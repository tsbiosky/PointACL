import numpy as np
import os
import sys
path = os.getcwd()
sys.path.append(path)
import open3d as o3d
import pandas as pd



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, './data/modelnet_pesudo.npz')
path2=os.path.join(BASE_DIR, './data/hf.npz')
td = np.load(path)
hf=np.load(path2)
ori_pc2=hf['ori_pc']
hf=hf['hf']
ori_pc = td["ori_pc"]
label=td["label40"]
ftt=[]
for iter in range(ori_pc.shape[0]):
    num_voxel=50
    #iter=2232
    if iter%100==0:
        print(iter)
    pc=ori_pc[iter]
    #print(pc)
    low=pc.min()
    high=pc.max()
    xrange=(low,high)
    yrange=(low,high)
    zrange=(low,high)
    dx=(xrange[1]-xrange[0])/num_voxel
    dy=(yrange[1]-yrange[0])/num_voxel
    dz=(zrange[1]-zrange[0])/num_voxel
    #print(xrange,yrange,zrange)
    vv=np.zeros((num_voxel+1,num_voxel+1,num_voxel+1),dtype=np.int)
    coord=np.empty((num_voxel+1,num_voxel+1,num_voxel+1),dtype='object')
    dd={}
    for i in range(num_voxel+1):
        for j in range(num_voxel + 1):
            for k in range(num_voxel + 1):
                key=str(i)+str(j)+str(k)
                dd[key]=[(i+0.5)*dx+xrange[0],(j+0.5)*dy+yrange[0],(k+0.5)*dz+zrange[0]]
                coord[i,j,k]=str(key)
    coord=coord.reshape(-1)
    try:
        for i in range(pc.shape[0]):
            x=int((pc[i,0]-(xrange[0]))/dx)
            y=int((pc[i,1]-(yrange[0]))/dy)
            z=int((pc[i,2]-(zrange[0]))/dz)
            vv[x,y,z]+=1
    except:
        print('error'+str(iter))
    f=np.fft.fftn(vv)
    fs=np.fft.fftshift(f)
    mid=int(num_voxel/2)
    fs[ mid- 20:mid + 20, mid - 20:mid + 20,mid - 20:mid + 20] = 0
    f_ishift = np.fft.ifftshift(fs)
    vv_back = np.fft.ifftn(f_ishift)
    vv_back=np.abs(vv_back)
    ind=np.argsort(vv_back,axis=None)
    select=ind[-1-128:-1]
    coord=coord[select]
    re=[]
    for i in range(128):
        re.append(dd[coord[i]])
    re=np.asarray(re)
    ftt.append(re)
ftt=np.asarray(ftt)
print(ftt.shape)
path = os.path.join(BASE_DIR, './data/ftt.npz')
np.savez(path, ori_pc=ori_pc.astype('float32'), ftt=ftt.astype('float32'),label=label.astype('int64'))
'''
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([point_cloud])
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points = o3d.utility.Vector3dVector(re)
o3d.visualization.draw_geometries([point_cloud2])

point_cloud3 = o3d.geometry.PointCloud()
point_cloud3.points = o3d.utility.Vector3dVector(hf_sample)
o3d.visualization.draw_geometries([point_cloud3])
#print(ind)


'''
