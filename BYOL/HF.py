import numpy as np
import os
import sys
import torch
path = os.getcwd()
sys.path.append(path)
import open3d as o3d
from torch.utils.data import DataLoader
from BYOL.data.data import ModelNet40,ModelNet10
import pcl
from numpy import linalg as LA
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
'''
torch.multiprocessing.set_sharing_strategy('file_system')
train_loader = DataLoader(
        ModelNet40_pesudo(partition='train', num_points=2048, normalize=True,transform=None))
ori_pc=[]

label_l=[]
for batch_idx, (pc, label) in enumerate(train_loader):
    ori_pc.append(pc)
    label_l.append(label)
ori_pc = np.concatenate(ori_pc, axis=0)
label_l=np.concatenate(label_l, axis=0)
'''
cloud=pcl.PointCloud()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, './data/hf.npz')
td = np.load(path)
for i in td.keys():
    print(i)
ori_pc = td["ori_pc"]
#label=td["label40"]
#for i in range(ori_pc.shape[0]):
hf=[]
c=0.75
N=int(2048*c)
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
    hf.append([ori_pc[i][index[-N-1:-1]]])
hf=np.concatenate(hf, axis=0)
print(hf.shape)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, './data/hf-0.75.npz')
np.savez(path, ori_pc=ori_pc.astype('float32'), hf=hf.astype('float32'))
#print(normal.size)
#print(ori_pc[0].shape)