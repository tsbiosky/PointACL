import numpy as np
import open3d as o3d
from numpy import linalg as LA
'''
path='./BYOL/data/hf-vis-0.5.npz'
npz = np.load(path, allow_pickle=True)

points = npz['ori_pc']
hd=npz['hf'][4]
nromal=points[4]
path='/home/junxuan/Desktop/PointFlowRenderer/vis.npz'
np.savez(path, ori_pc=nromal.astype('float32'), hd=hd.astype('float32'))

def don(ori):
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
'''
#path='./BYOL/data/PointACL.npz'
path='./BYOL/data/combine.npz'
td = np.load(path, allow_pickle=True)
aug=td['aug']
adv=td['adv']
adv_rocl=td['adv_rocl']
adv_acl=td['adv_acl']
adv_sup=td['adv_sup']

#hd=td['hf']
#pertu=td['pert']
aug=aug.transpose(0,2,1)
adv=adv.transpose(0,2,1)
adv_rocl=adv_rocl.transpose(0,2,1)
adv_acl=adv_acl.transpose(0,2,1)
adv_sup=adv_sup.transpose(0,2,1)
#pertu=pertu.transpose(0,2,1)
#print(aug.shape,adv.shape)
#pertu=abs(pertu)
label=td['label']


for i in range(1):
    i=310
    if label[i]!=0:
        continue
    print(i)
    point_cloud = o3d.geometry.PointCloud()
    #points=points.transpose(0,2,1)
    point_cloud.points = o3d.utility.Vector3dVector(adv_sup[i])
    o3d.visualization.draw_geometries([point_cloud])
    #
    #points2 = npz['hf']
    #point_cloud2 = o3d.geometry.PointCloud()
    #points2=points2.transpose(0,2,1)
    #point_cloud2.points = o3d.utility.Vector3dVector(adv[i])
    #o3d.visualization.draw_geometries([point_cloud2])
print(i)
pertu=abs(aug-adv_sup)
dis=pertu[i,:,0]**2+pertu[i,:,1]**2+pertu[i,:,2]**2
print(dis.shape)
index = np.argsort(dis, axis=0)
index=index[-50::]
'''
aug[i][index]=[0,0,0]
point_cloud3 = o3d.geometry.PointCloud()
#points2=points2.transpose(0,2,1)
point_cloud3.points = o3d.utility.Vector3dVector(aug[i])
o3d.visualization.draw_geometries([point_cloud3])
'''

path='/home/junxuan/Desktop/PointFlowRenderer/vis_adv.npz'
np.savez(path, aug=aug[i].astype('float32'), adv=adv_sup[i].astype('float32'),index=index)
'''
points3 = npz['aug2']
point_cloud3 = o3d.geometry.PointCloud()
points3=points3.transpose(0,2,1)
point_cloud3.points = o3d.utility.Vector3dVector(points3[i])
o3d.visualization.draw_geometries([point_cloud3])
'''