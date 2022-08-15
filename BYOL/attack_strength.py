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
def linear_train(epoch, model, Linear, loptim, trainloader,attacker=None):


    return acc, model, Linear, loptim


def adjust_lr(epoch, optim):
    lr = 0.2
    lr_list = [30, 50, 100]
    if epoch >= lr_list[0]:
        lr = lr / 10
    if epoch >= lr_list[1]:
        lr = lr / 10
    if epoch >= lr_list[2]:
        lr = lr / 10

    for param_group in optim.param_groups:
        param_group['lr'] = lr


def test(model,Linear,testloader,attacker,args):
    model.eval()
    Linear.eval()

    test_clean_loss = 0
    test_adv_loss = 0
    clean_correct = 0
    adv_correct = 0
    target_correct=0
    clean_acc = 0
    total = 0
    ori_npz=[]
    adv_npz=[]
    for idx, (pc, label) in enumerate(testloader):
        with torch.no_grad():
            pc, y = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True).squeeze()
            #target = target.long().cuda(non_blocking=True)
        adv_inputs, success_num = attacker.attack(pc, y)
        total += y.size(0)
        pc = pc.permute(0, 2, 1)
        feat,_=model(pc)
        out = Linear(feat)
        _, predx = torch.max(out.data, 1)
        criterion = nn.CrossEntropyLoss()
        clean_loss = criterion(out, y)
        clean_correct += predx.eq(y.data).cpu().sum().item()

        clean_acc = 100. * clean_correct / total

        test_clean_loss += clean_loss.data
        feat,_= model(adv_inputs)
        out = Linear(feat)

        _, predx = torch.max(out.data, 1)
        adv_loss = criterion(out, y)

        adv_correct += predx.eq(y.data).cpu().sum().item()
        adv_acc = 100. * adv_correct / total

        ngpus_per_node = torch.cuda.device_count()
        test_adv_loss += adv_loss.data
        '''
        if args.local_rank % ngpus_per_node == 0:
            progress_bar(idx, len(testloader),
                         'Testing Loss {:.3f}, acc {:.3f} , adv Loss {:.3f}, adv acc {:.3f},target acc {:.3f}'.format(
                             test_clean_loss / (idx + 1), clean_acc, test_adv_loss / (idx + 1), adv_acc,target_acc))
        '''
        ori_npz.append(pc.detach().cpu().numpy())
        adv_npz.append(adv_inputs.detach().cpu().numpy())
    print("Test accuracy: {0}".format( adv_acc))
    ori_npz=np.concatenate(ori_npz, axis=0)
    adv_npz=np.concatenate(adv_npz, axis=0)
    #np.savez('./adv_test.npz', ori=ori_npz.astype('float32'), adv=adv_npz.astype('float32'))
    return (clean_acc, adv_acc)
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
def NT_xent(out_1,out_2,out3=None,temperature=0.5):
    outputs_1=out_1
    outputs_2=out_2
    if out3==None:
        outputs = torch.cat((outputs_1, outputs_2))
    else:
        outputs = torch.cat((outputs_1, outputs_2,out3))
    B = outputs.shape[0]
    outputs_norm = outputs / (outputs.norm(dim=1).view(B, 1) + 1e-8)
    similarity_matrix = (1. / temperature) * torch.mm(outputs_norm, outputs_norm.transpose(0, 1).detach())
    N2 = len(similarity_matrix)
    if out3!=None:
        N = int(len(similarity_matrix) / 3)
    else:
        N=int(len(similarity_matrix) / 2)
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2, N2)).cuda()

    NT_xent_loss = - torch.log(
        similarity_matrix_exp / (torch.sum(similarity_matrix_exp, dim=1).view(N2, 1) + 1e-8) + 1e-8)
    if out3!=None:
        NT_xent_loss_total = (1. / float(N2)) * torch.sum(
            torch.diag(NT_xent_loss[0:N, N:2 * N]) + torch.diag(NT_xent_loss[N:2 * N, 0:N])
            + torch.diag(NT_xent_loss[0:N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, 0:N])
            + torch.diag(NT_xent_loss[N:2 * N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, N:2 * N]))
    else:
        NT_xent_loss_total = (1. / float(N2)) * torch.sum(
            torch.diag(NT_xent_loss[0:N, N:]) + torch.diag(NT_xent_loss[N:, 0:N]))
    return NT_xent_loss_total
def get_norm(x):
    """Calculate the norm of a given data x.

    Args:
        x (torch.FloatTensor): [B, 3, K]
    """
    # use global l2 norm here!
    norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
    return norm
class Head(nn.Module):
    def __init__(self,emb_dims,output_channels):
        super(Head, self).__init__()
        self.linear1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)
    def forward(self,x):
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
def get_emd(model,Linear,trainloader,attacker):
    path=('/mnt/ssd/home/junxuan/saved_pc.npz')
    td=np.load(path)
    aug=td['aug']
    adv=td['adv']
    adv_npz = []
    model.eval()
    Linear.eval()
    for idx, (inputs,inputs2,label)  in enumerate(trainloader):
        with torch.no_grad():
            pc, y = inputs.float().cuda(non_blocking=True), \
                    label.long().cuda(non_blocking=True).squeeze()
        adv_inputs, success_num = attacker.attack(pc, y)
        adv_inputs=adv_inputs.detach().cpu().numpy()
        adv_npz.append(adv_inputs)
    adv_npz = np.concatenate(adv_npz, axis=0)
    print(adv_npz.shape)
    np.savez('/mnt/ssd/home/junxuan/saved_pc_withSUP.npz', unsup=adv.astype('float32'), sup=adv_npz.astype('float32'))


if __name__ == "__main__":
    torch.cuda.set_device(1)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", type=str, help="the ckpt path to load")
    parser.add_argument("--xyz_only", default=1, type=str, help="whether to only use xyz-coordinate for evaluation")
    parser.add_argument("--num_points", default=2048, type=int)
    parser.add_argument("--k", default=40, type=int, help="choose gpu")
    parser.add_argument("--dropout", default=0.5, type=float, help="choose gpu")
    parser.add_argument("--emb_dims", default=1024, type=int, help="dimension of hidden embedding")
    parser.add_argument("--batch_size", default=12, type=int, help="batch size of dataloader")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudUpSampling(max_num_points=args.num_points * 2, centroid="random"),
            d_utils.PointcloudRandomCrop(p=0.5, min_num_points=args.num_points),
            d_utils.PointcloudNormalize(),
            d_utils.PointcloudRandomCutout(p=0.5, min_num_points=args.num_points),
            d_utils.PointcloudScale(p=1),
            # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
            d_utils.PointcloudRotatePerturbation(p=1),
            d_utils.PointcloudTranslate(p=1),
            d_utils.PointcloudJitter(p=1),
            d_utils.PointcloudRandomInputDropout(p=1),
            # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
        ]
    )


    # hparam = load_hparam(args.model_yaml)
    hparam = {"model.use_xyz": True, "emb_dims": args.emb_dims, "dropout":args.dropout, "num_points": args.num_points, "k": args.k,"mlp_hidden_size":4096,"projection_size":256}
    #Linear = Head(args.emb_dims,40)
    Linear_rocl=nn.Sequential(nn.Linear(args.emb_dims, 40))
    #model = PointNet(hparam)
    model_rocl=TargetNetwork_PointNet(hparam)
    path_rocl='/mnt/ssd/home/junxuan/header/ROCL.ckpt'
    model_rocl= load_model(path_rocl, model_rocl)
    Linear_rocl.load_state_dict(torch.load('/mnt/ssd/home/junxuan/header/ROCL_linear.pth'))
    Linear_rocl.eval()
    model_rocl.eval()
    model_rocl.cuda()
    Linear_rocl.cuda()


    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points, normalize=False,transform=train_transforms),
                              num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, normalize=False),
                             num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    #train_sampler = DistributedSampler(train_dataset, shuffle=False)


    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    #test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False
                             )
    adv_func = CrossEntropyAdvLoss()
    ###attack setting
    delta = args.budget
    args.budget = args.budget * \
                  np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
    args.num_iter = int(args.num_iter)
    args.step_size = args.budget / float(args.num_iter)
    clip_func = ClipPointsL2(budget=args.budget)
    #clip_func=ClipPointsLinf(budget=args.budget)
    #Linear.load_state_dict(torch.load('./best_linear_STRL.pth'))
    ##train

    attacker_test1 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                         clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                         num_iter=1, dist_metric='l2')

    attacker_test2 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                         clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                         num_iter=3, dist_metric='l2')
    attacker_test3 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                          clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                          num_iter=5, dist_metric='l2')
    attacker_test4 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                          clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                          num_iter=7, dist_metric='l2')
    attacker_test5 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                          clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                          num_iter=10, dist_metric='l2')
    attacker_test6 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                          clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                          num_iter=15, dist_metric='l2')
    attacker_test7 = IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                          clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                          num_iter=20, dist_metric='l2')
    step=[1,3,5,7,10,15,20,25,30,35]
    re=[]
    for i in range(20):
        delta = 0.001*(i+1)
        args.budget = delta * \
                      np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
        args.num_iter = 5
        args.step_size = args.budget / float(args.num_iter)
        clip_func = ClipPointsL2(budget=args.budget)
        #print(delta,args.budget)
        attacker_test= IFGM(model_rocl, linear=Linear_rocl, adv_func=adv_func,
                              clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                              num_iter=args.num_iter, dist_metric='l2')
        clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test, args)
        re.append(adv_acc)
    path ='/mnt/ssd/home/junxuan/ROCL.npz'
    np.savez(path,acc=re)
    '''
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test1, args)
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test2, args)
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test3, args)
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test4, args)
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test5, args)
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test6, args)
    clean_acc, adv_acc = test(model_rocl, Linear_rocl, val_loader, attacker_test7, args)
    '''