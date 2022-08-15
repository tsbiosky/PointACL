#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import random
from torch.utils.data import Dataset
from copy import deepcopy
from data.data_utils import points_sampler

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR=BASE_DIR
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    #print(all_data.shape)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', normalize=False,transform=None,kmeans=None):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        self.kmeans=kmeans
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        ori=deepcopy(pointcloud)
        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        if self.kmeans==1:
            return pointcloud,label,ori
        else:
            if self.transform!=None:
                pointcloud2 = deepcopy(pointcloud)
                point_set = self.transform(pointcloud)
                point_set_ = self.transform(pointcloud2)
                point_set = points_sampler(point_set, self.num_points)
                point_set_ = points_sampler(point_set_, self.num_points)
                return point_set, point_set_, label
            else:
                return pointcloud, label
    def __len__(self):
        return self.data.shape[0]
class ModelNet40npz(Dataset):
    def __init__(self, num_points, partition='train', normalize=False,transform=None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.partition = partition
        if self.partition=='train':
            self.path = os.path.join(BASE_DIR, 'ori.npz')
        else:
            self.path = os.path.join(BASE_DIR, 'ori_test.npz')
        self.td = np.load(self.path)
        self.data, self.label = self.td['data'],self.td['label']
        self.num_points = num_points
        self.normalize = normalize
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        ori=deepcopy(pointcloud)
        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        if self.transform != None:
            pointcloud2 = deepcopy(pointcloud)
            point_set = self.transform(pointcloud)
            point_set_ = self.transform(pointcloud2)
            point_set = points_sampler(point_set, self.num_points)
            point_set_ = points_sampler(point_set_, self.num_points)
            return point_set, point_set_, label
        else:
            return pointcloud, label
    def __len__(self):
        return self.data.shape[0]
class ModelNet10(Dataset):
    def __init__(self, num_points, partition='train', normalize=False,transform=None):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        self.top10=[0,2,4,5,8,22,30,33,35,37]
        index_list=[]
        for i in range(self.data.shape[0]):
            if self.label[i] in self.top10:
                index_list.append(i)
        self.data=self.data[index_list]
        self.label=self.label[index_list]
        for i in range(self.label.shape[0]):
            self.label[i]=self.top10.index(self.label[i])
        print(len(index_list),self.label.shape[0])
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        if self.transform!=None:
            pointcloud2 = deepcopy(pointcloud)
            point_set = self.transform(pointcloud)
            point_set_ = self.transform(pointcloud2)
            point_set = points_sampler(point_set, self.num_points)
            point_set_ = points_sampler(point_set_, self.num_points)
            return point_set, point_set_, label
        else:
            return pointcloud, label
    def __len__(self):
        return self.data.shape[0]

class ModelNet40SSL(Dataset):
    def __init__(self, num_points, partition='train', normalize=False,transform=None):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        pointcloud2 = deepcopy(pointcloud)
        point_set = self.transform(pointcloud)
        point_set_ = self.transform(pointcloud2)
        point_set = points_sampler(point_set, self.num_points)
        point_set_ = points_sampler(point_set_, self.num_points)
        return point_set, point_set_
    def __len__(self):
        return self.data.shape[0]
class ModelNet40attack(Dataset):
    def __init__(self, num_points, partition='train', normalize=False,path=''):
        #self.data, self.label = load_data(partition)
        self.path = path
        self.td = np.load(self.path)
        self.data = self.td["data"]
        self.label=self.td["label"]
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
class ModelNet40_hg(Dataset):
    def __init__(self, num_points, partition='train', normalize=False,path='./hf.npz',transform=None):
        #self.data, self.label = load_data(partition)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(BASE_DIR, path)
        self.td = np.load(self.path)
        self.data = self.td["ori_pc"]
        self.label=self.td["label"]
        #self.ftt=self.td["ftt"]
        self.hf=self.td["hf"]
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        hf=self.hf[item]
        #ftt=self.ftt[item]
        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        if self.transform==None:
            return pointcloud,label
        pointcloud2 = deepcopy(pointcloud)
        point_set = self.transform(pointcloud)
        point_set_ = self.transform(pointcloud2)
        point_set = points_sampler(point_set, self.num_points)
        point_set_ = points_sampler(point_set_, self.num_points)
        return point_set, point_set_,hf,label


    def __len__(self):
        return self.data.shape[0]
class ModelNet40Subset(Dataset):
    def __init__(self, num_points, partition='train', normalize=False, percent=1.0):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        self.percent = percent

        if self.percent < 1:
            self.data, self.label = self.sample_data()
            print("Remain data and label:", self.data.shape, self.label.shape)

    def sample_data(self):
        data_by_label = {}
        for i in range(len(self.data)):
            label = self.label[i][0]
            if label not in data_by_label:
                data_by_label[label] = [self.data[i]]
            else:
                data_by_label[label].append(self.data[i])
        chosen_data = []
        chosen_label = []
        all_data = []
        all_label = []
        for label in data_by_label:
            idx = list(range(len(data_by_label[label])))
            cidx = np.random.choice(idx)
            chosen_data.append(data_by_label[label][cidx])
            chosen_label.append(label)
            del data_by_label[label][cidx]
            all_data.extend(data_by_label[label])
            all_label.extend([label]*len(data_by_label[label]))
        remain_num = int(round(len(self.data) * self.percent)) - len(chosen_data)
        idx = list(range(len(all_data)))
        cidx = random.sample(idx, remain_num)
        chosen_data = np.array(chosen_data)
        chosen_label = np.array(chosen_label)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        chosen_data = np.concatenate([chosen_data, all_data[cidx]], 0)
        chosen_label = np.concatenate([chosen_label, all_label[cidx]], 0)
        chosen_label = chosen_label.reshape(-1, 1)
        return chosen_data, chosen_label

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.normalize:
            pointcloud[:, :3] = pc_normalize(pointcloud[:, :3])

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



if __name__ == '__main__':
    train = ModelNet40Subset(1024, percent=0.01)
    test = ModelNet40Subset(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
