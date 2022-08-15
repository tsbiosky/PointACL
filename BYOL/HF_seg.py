import numpy as np
import os
import sys
import torch
path = os.getcwd()
sys.path.append(path)
import open3d as o3d
from torch.utils.data import DataLoader
from BYOL.data.S3DIS import  S3DIS, load_data_semseg
import pcl
from numpy import linalg as LA
import argparse
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from BYOL.data.S3DIS import  S3DIS
import numpy as np
from torch.utils.data import DataLoader
from util.adv_utils import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from ROCL.FGM import IFGM_seg,PGD
from BYOL.models.networks_dgcnn_semseg import DGCNN_semseg,DGCNN_head
#from BYOL.models.dgcnn_pure import DGCNN_semseg
from dataset import ModelNet40Attack
from util.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss
from util.clip_utils import ClipPointsL2,ClipPointsLinf
global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
def one(n):
    n=np.asarray(n)
    l=LA.norm(n)
    for i in range(4):
        n[i]=n[i]/l
    return n
def convert(normal):
    n=normal.size
    re=[]
    for i in range(n):
        re.append(one(normal[i]))
    return re
def get_norm( x):
    """Calculate the norm of a given data x.

    Args:
        x (torch.FloatTensor): [B, 3, K]
    """
    # use global l2 norm here!
    norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
    return norm
parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['dgcnn'],
                    help='Model to use, [dgcnn]')
parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                    choices=['S3DIS'])
parser.add_argument('--test_area', type=str, default='6', metavar='N',
                    choices=['1', '2', '3', '4', '5', '6', 'all'])
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=4096,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_root', type=str, default='', metavar='N',
                    help='Pretrained model root')
parser.add_argument('--visu', type=str, default='',
                    help='visualize the model')
parser.add_argument('--visu_format', type=str, default='ply',
                    help='file format of visualization')
parser.add_argument("--finetune", default=False, type=bool, help='finetune the model')
parser.add_argument("-w", "--weight", type=str, help="the ckpt path to load")
parser.add_argument('--budget', type=float, default=0.08,
                    help='FGM attack budget')
parser.add_argument('--num_iter', type=int, default=7,
                    help='IFGM iterate step')
args = parser.parse_args()
ori_pc, label = load_data_semseg('train', '6')
torch.cuda.set_device(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
ori_pc=ori_pc[:,:,0:3]
pc=ori_pc[0]
low=pc.min()
high=pc.max()
print(low,high)
cloud=pcl.PointCloud()
train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area),
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
hparam = {"model.use_xyz": True, "emb_dims": 1024, "dropout": 0.5, "num_points": 4096,
              "k": 20, "mlp_hidden_size": 4096, "projection_size": 256}
model = DGCNN_semseg(hparam,finetune=True)
model = nn.DataParallel(model,device_ids=[1,2,3,4])
print("Let's use", torch.cuda.device_count(), "GPUs!")
device = torch.device("cuda")
model.to(device)
head= DGCNN_head(hparam)
#model.cuda()
head.cuda()
model_params = []
if args.finetune:
    model_params += model.parameters()
model_params += head.parameters()
if args.use_sgd:
    print("Use SGD")
    opt = optim.SGD(model_params, lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
else:
    print("Use Adam")
    opt = optim.Adam(model_params, lr=args.lr, weight_decay=1e-4)

if args.scheduler == 'cos':
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
elif args.scheduler == 'step':
    scheduler = StepLR(opt, 20, 0.5, args.epochs)
criterion = cal_loss
#adv_func = CrossEntropyAdvLoss()
###attack setting
delta = args.budget
args.budget = args.budget * \
              np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
args.num_iter = int(args.num_iter)
args.step_size = args.budget / float(args.num_iter)
clip_func = ClipPointsL2(budget=args.budget)
attacker_test = IFGM_seg(model, linear=head, adv_func=cal_loss,
                     clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                     num_iter=args.num_iter, dist_metric='l2')
best_test_iou = 0
BASE_DIR = '/mnt/ssd/home/junxuan'
DATA_DIR = os.path.join(BASE_DIR, 'data')
path = os.path.join(BASE_DIR, './data/hf_S3DIS.npz')
'''
for epoch in range(args.epochs):
    ####################
    # Train
    ####################
    train_loss = 0.0
    count = 0.0
    #model.train()
    if args.finetune:
        model.train()
    else:
        model.eval()
    head.train()
    train_true_cls = []
    train_pred_cls = []
    train_true_seg = []
    train_pred_seg = []
    train_label_seg = []
    saved_adv = []
    for data, seg in train_loader:
        data, seg = data.cuda(), seg.cuda()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        opt.zero_grad()
        ###
        adv_data = data.clone()
        x1, x2, x3, x = model(data)
        seg_pred = head(x1, x2, x3, x)
        # seg_pred = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        with torch.enable_grad():
            for i in range(args.num_iter):
                adv_data.requires_grad_(True)
                x1, x2, x3, x = model(adv_data)
                seg_pred_adv = head(x1, x2, x3, x)
                # seg_pred = model(data)
                seg_pred_adv = seg_pred_adv.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred_adv.view(-1, 13), seg.view(-1, 1).squeeze())
                ###
                grad_outputs = None
                grad = torch.autograd.grad(loss, adv_data, grad_outputs=grad_outputs, only_inputs=True, retain_graph=True)[0]
                norm = get_norm(grad)
                normalized_grad = grad / (norm[:, None, None] + 1e-9)
                perturbation = args.step_size * normalized_grad
                sorted, indices = torch.sort(perturbation, dim=-1,descending=True)
                indices=indices[:,:,0:512]
                selected=torch.gather(data,2,indices)
                #print(selected.shape)
                # add perturbation and clip
                adv_data = adv_data + perturbation
                adv_data = clip_func(adv_data, data)
            #selected=selected.permute(0,2,1)
        if epoch>10:
            selected=selected.detach().cpu().numpy()
            np.savez(path, ori_pc=data.astype('float32'), hf=selected.astype('float32'))
        loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        loss.backward()
        opt.step()
        pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
        count += batch_size
        train_loss += loss.item() * batch_size
        seg_np = seg.cpu().numpy()  # (batch_size, num_points)
        pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
        train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
        train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
        train_true_seg.append(seg_np)
        train_pred_seg.append(pred_np)
    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 1e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 1e-5
    train_true_cls = np.concatenate(train_true_cls)
    train_pred_cls = np.concatenate(train_pred_cls)
    train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
    train_true_seg = np.concatenate(train_true_seg, axis=0)
    train_pred_seg = np.concatenate(train_pred_seg, axis=0)
    train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
    outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                              train_loss * 1.0 / count,
                                                                                              train_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(train_ious))
    io.cprint(outstr)
'''
#for i in range(ori_pc.shape[0]):
hf=[]
for i in range(ori_pc.shape[0]):
    if i%100==0:
        print(i)
    cloud.from_array(ori_pc[i])
    ne1=cloud.make_NormalEstimation()
    tree=cloud.make_kdtree()
    ne1.set_SearchMethod(tree)
    ne1.set_RadiusSearch(0.05)
    n_s=ne1.compute()
    n_s=convert(n_s)##list
    ne2=cloud.make_NormalEstimation()
    ne2.set_SearchMethod(tree)
    ne2.set_RadiusSearch(0.2)
    n_l=ne2.compute()
    n_l=convert(n_l)##list
    diff=np.zeros((len(n_s)))
    for ii in range(len(n_s)):
        t=np.zeros((4))
        for j in range(4):
            t[j]=n_s[ii][j]-n_l[ii][j]
        diff[ii]=LA.norm(t)
    index=np.argsort(diff)
    hf.append([ori_pc[i][index[-3072-1:-1]]])
hf=np.concatenate(hf, axis=0)
print(hf.shape)
np.savez(path, ori_pc=ori_pc.astype('float32'), hf=hf.astype('float32'),label=label.astype('int64'))
#print(normal.size)
