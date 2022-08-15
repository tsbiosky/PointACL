import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = os.getcwd()
sys.path.append(path)

import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import SVC
import torch.nn as nn

from BYOL.models import PointNet
from BYOL.models.networks_PointNet import TargetNetwork_PointNet, OnlineNetwork_PointNet
from BYOL.data import ModelNet40Cls
from BYOL.data.data import ModelNet40,ModelNet10
from ROCL.utils import progress_bar, checkpoint
from ROCL.FGM import IFGM,PGD
from dataset import ModelNet40Attack
from util.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss
from util.clip_utils import ClipPointsL2,ClipPointsLinf
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision import transforms
import BYOL.data.data_utils as d_utils
from copy import deepcopy
from sklearn.cluster import KMeans
def load_model(weight_path, model):
    state_dict = model.state_dict()

    ckpt = torch.load(weight_path, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if "online_network." + key in pretrained_dict:
            state_dict[key] = pretrained_dict["online_network."+key]
            print(key)

    model.load_state_dict(state_dict, strict=True)
    return model

def plot_k_means(x, r, k):
    random_colors = np.random.random((k, 3))
    colors = r.dot(random_colors)
    print(r[:20])
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.savefig('kmeans.png')


def initialize_centers(x, num_k):
    N, D = x.shape
    centers = np.zeros((num_k, D))
    used_idx = []
    for k in range(num_k):
        idx = np.random.choice(N)
        while idx in used_idx:
            idx = np.random.choice(N)
        used_idx.append(idx)
        centers[k] = x[idx]
    return centers


def update_centers(x, r, K):
    N, D = x.shape
    centers = np.zeros((K, D))
    for k in range(K):
        centers[k] = r[:, k].dot(x) / r[:, k].sum()
    return centers


def square_dist(a, b):
    return (a - b) ** 2


def cost_func(x, r, centers, K):
    cost = 0
    for k in range(K):
        norm = np.linalg.norm(x - centers[k], 2)
        cost += (norm * np.expand_dims(r[:, k], axis=1)).sum()
    return cost


def cluster_responsibilities(centers, x, beta):
    N, _ = x.shape
    K, D = centers.shape
    R = np.zeros((N, K))

    for n in range(N):
        R[n] = np.exp(-beta * np.linalg.norm(centers - x[n], 2, axis=1))
    R /= R.sum(axis=1, keepdims=True)

    return R


def soft_k_means(x, K, max_iters=300, beta=1.):
    centers = initialize_centers(x, K)
    prev_cost = 0
    for i in range(max_iters):
        print(i)
        r = cluster_responsibilities(centers, x, beta)
        centers = update_centers(x, r, K)
        cost = cost_func(x, r, centers, K)
        if np.abs(cost - prev_cost) < 1e-5:
            break
        prev_cost = cost
    return r

def generate_samples(std=1, dim=2, dist=4):
    mu0 = np.array([0, 0])
    mu1 = np.array([dist, dist])
    mu2 = np.array([0, dist])
    # num samps per class
    Nc = 300
    x0 = np.random.randn(Nc, dim) * std + mu0
    x1 = np.random.randn(Nc, dim) * std + mu1
    x2 = np.random.randn(Nc, dim) * std + mu2
    x = np.concatenate((x0, x1, x2), axis=0)
    return x


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", type=str, help="the ckpt path to load")
    parser.add_argument("--xyz_only", default=1, type=str, help="whether to only use xyz-coordinate for evaluation")
    parser.add_argument("--num_points", default=2048, type=int)
    parser.add_argument("--k", default=40, type=int, help="choose gpu")
    parser.add_argument("--dropout", default=0.5, type=float, help="choose gpu")
    parser.add_argument("--emb_dims", default=1024, type=int, help="dimension of hidden embedding")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size of dataloader")
    parser.add_argument("--gpu_num", default=0, type=int, help="choose gpu")
    parser.add_argument("--finetune", default=False, type=bool, help='finetune the model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int,
                        help='total epochs to run')
    parser.add_argument('--data_root', type=str,
                        default='dataset/attack_data.npz')
    parser.add_argument('--budget', type=float, default=0.01,
                        help='FGM attack budget')
    parser.add_argument('--num_iter', type=int, default=7,
                        help='IFGM iterate step')
    parser.add_argument('--mu', type=float, default=1.,
                        help='momentum factor for MIFGM attack')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # hparam = load_hparam(args.model_yaml)
    hparam = {"model.use_xyz": True, "emb_dims": args.emb_dims, "dropout": args.dropout, "num_points": args.num_points,
              "k": args.k, "mlp_hidden_size": 4096, "projection_size": 256}
    model = TargetNetwork_PointNet(hparam)
    model = load_model(args.weight, model)
    model.cuda()
    model_params = []
    cudnn.benchmark = True
    model.eval()
    train_loader = DataLoader(
        ModelNet40(partition='train', num_points=args.num_points, normalize=False, transform=None, kmeans=1),
        num_workers=8,
        batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, normalize=False),
                            num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=False)
    ori_pc = []
    feat = []
    label_l = []
    for batch_idx, (pc, label, ori) in enumerate(train_loader):
        pc, label = pc.cuda(), label.squeeze()
        pc = pc.permute(0, 2, 1)
        y1, z1 = model(pc)
        ori_pc.append(ori)
        label_l.append(label)
        feat.append(y1.detach().cpu().numpy())
    ori_pc = np.concatenate(ori_pc, axis=0)
    label_l = np.concatenate(label_l, axis=0)
    feat = np.concatenate(feat, axis=0)
    pred=soft_k_means(feat, K=40)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, './data/modelnet_pesudo_soft.npz')
    np.savez(path, ori_pc=ori_pc.astype('float32'), label=pred.astype('int64'))
    #plot_k_means(feat, pred, 40)

if __name__ == "__main__":
    main()