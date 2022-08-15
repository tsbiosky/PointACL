import os
import sys

path = os.getcwd()
sys.path.append(path)
from util.losses import SinkhornDistance
from sklearn.datasets import make_moons
import torch
import numpy as np
torch.cuda.set_device(3)
path='/mnt/ssd/home/junxuan/saved_pc_withSUP.npz'
td = np.load(path)
unsup=td['unsup']
sup=td['unsup']
print(unsup.shape)
unsup=np.transpose(unsup, (0, 2, 1))
sup=np.transpose(sup, (0, 2, 1))
#X, Y = make_moons(n_samples = 30)
#a = X[Y==0]
#b = X[Y==1]

x = torch.tensor(unsup, dtype=torch.float).cuda()
y = torch.tensor(sup, dtype=torch.float).cuda()

sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
sample=64
dist, P, C = sinkhorn(x[-1-64:-1], y[-65:-1])
print(dist)