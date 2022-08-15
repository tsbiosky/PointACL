import numpy as np
import open3d as o3d
from numpy import linalg as LA
import pcl
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
def don(ori):
    N=int(2048*0.5)
    cloud = pcl.PointCloud()
    cloud.from_array(ori)
    ne1 = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne1.set_SearchMethod(tree)
    ne1.set_RadiusSearch(0.05)
    n_s = ne1.compute()
    n_s = convert(n_s)  ##list
    ne2 = cloud.make_NormalEstimation()
    ne2.set_SearchMethod(tree)
    ne2.set_RadiusSearch(0.2)
    n_l = ne2.compute()
    n_l = convert(n_l)  ##list
    diff = np.zeros((len(n_s)))
    for ii in range(len(n_s)):
        t = np.zeros((4))
        for j in range(4):
            t[j] = n_s[ii][j] - n_l[ii][j]
        diff[ii] = LA.norm(t)
    index = np.argsort(diff)
    hd=ori[index[-N-1:-1]]
    return hd

path='./BYOL/data/vis_adv.npz'
td = np.load(path, allow_pickle=True)
aug=td['aug']
adv=td['adv']
print(adv.shape)
#aug=aug.transpose(0,2,1)
#adv=adv.transpose(0,2,1)
hd=don(aug)

path='/mnt/ssd/home/junxuan/vis/hd.npz'
np.savez(path, hd=hd.astype('float32'))
'''
points3 = npz['aug2']
point_cloud3 = o3d.geometry.PointCloud()
points3=points3.transpose(0,2,1)
point_cloud3.points = o3d.utility.Vector3dVector(points3[i])
o3d.visualization.draw_geometries([point_cloud3])
'''