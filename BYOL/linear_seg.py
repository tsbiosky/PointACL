#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_semseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import sys
import argparse
import torch

path = os.getcwd()
sys.path.append(path)
import argparse
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from BYOL.data.S3DIS import  S3DIS
from ROCL.utils import progress_bar, checkpoint
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
visual_warning = True
def load_model(weight_path, model):
    state_dict = model.state_dict()

    ckpt = torch.load(weight_path, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if "target_network." + key in pretrained_dict:
            state_dict[key] = pretrained_dict["target_network."+key]
            #print(key)

    model.load_state_dict(state_dict, strict=True)
    return model
def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg.py outputs'+'/'+args.exp_name+'/'+'main_semseg.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(13):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all 


def visualization(visu, visu_format, test_choice, data, seg, pred, visual_file_index, semseg_colors):
    global room_seg, room_pred
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = [] 
        skip = False
        with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
            files = f.readlines()
            test_area = files[visual_file_index][5]
            roomname = files[visual_file_index][7:-1]
            if visual_file_index + 1 < len(files):
                roomname_next = files[visual_file_index+1][7:-1]
            else:
                roomname_next = ''
        if visu[0] != 'all':
            if len(visu) == 2:
                if visu[0] != 'area' or visu[1] != test_area:
                    skip = True 
                else:
                    visual_warning = False
            elif len(visu) == 4:
                if visu[0] != 'area' or visu[1] != test_area or visu[2] != roomname.split('_')[0] or visu[3] != roomname.split('_')[1]:
                    skip = True
                else:
                    visual_warning = False  
            else:
                skip = True
        elif test_choice !='all':
            skip = True
        else:
            visual_warning = False
        if skip:
            visual_file_index = visual_file_index + 1
        else:
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname)
            
            data = np.loadtxt('data/indoor3d_sem_seg_hdf5_data_test/raw_data3d/Area_'+test_area+'/'+roomname+'('+str(visual_file_index)+').txt')
            visual_file_index = visual_file_index + 1
            for j in range(0, data.shape[0]):
                RGB.append(semseg_colors[int(pred[i][j])])
                RGB_gt.append(semseg_colors[int(seg[i][j])])
            data = data[:,[1,2,0]]
            xyzRGB = np.concatenate((data, np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((data, np.array(RGB_gt)), axis=1)
            room_seg.append(seg[i].cpu().numpy())
            room_pred.append(pred[i].cpu().numpy()) 
            f = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt', "a")
            f_gt = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt', "a")
            np.savetxt(f, xyzRGB, fmt='%s', delimiter=' ') 
            np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
            
            if roomname != roomname_next:
                mIoU = np.mean(calculate_sem_IoU(np.array(room_pred), np.array(room_seg), visual=True))
                mIoU = str(round(mIoU, 4))
                room_pred = []
                room_seg = []
                if visu_format == 'ply':
                    filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_pred_'+mIoU+'.ply'
                    filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.ply'
                    xyzRGB = np.loadtxt('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt')
                    xyzRGB_gt = np.loadtxt('outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt')
                    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                    vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath)
                    print('PLY visualization file saved in', filepath)
                    vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_gt)
                    print('PLY visualization file saved in', filepath_gt)
                    os.system('rm -rf '+'outputs/'+args.exp_name+'/visualization/area_'+test_area+'/'+roomname+'/*.txt')
                else:
                    filename = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt'
                    filename_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt'
                    filename_mIoU = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_pred_'+mIoU+'.txt'
                    os.rename(filename, filename_mIoU)
                    print('TXT visualization file saved in', filename_mIoU)
                    print('TXT visualization file saved in', filename_gt)
            elif visu_format != 'ply' and visu_format != 'txt':
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            
        
def train(args, io):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,4"
    train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area), 
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    #device = torch.device(1)
    hparam = {"model.use_xyz": True, "emb_dims": 1024, "dropout": 0.5, "num_points": 4096,
              "k": 20, "mlp_hidden_size": 4096, "projection_size": 256}
    model = DGCNN_semseg(hparam,finetune=True)
    #model=DGCNN_semseg(hparam)
    if not args.finetune:
        model = load_model(args.weight, model)
    head= DGCNN_head(hparam)
    #model.cuda()
    head.cuda()
    #Try to load models
    print(len(train_loader))
    #device = torch.device("cuda")
    #model.to(device)
    model = nn.DataParallel(model,device_ids=[1,2,5,4])
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.finetune:
        model.train()
        print('finetune')
    else:
        model.eval()
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
    attacker_train = IFGM_seg(model, linear=head, adv_func=cal_loss,
                             clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                             num_iter=5, dist_metric='l2')
    best_test_iou = 0

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
        for batch_idx,(data, seg) in enumerate(train_loader):
            data, seg = data.cuda(), seg.cuda()

            batch_size = data.size()[0]
            opt.zero_grad()
            #input_total=data
            if args.finetune:
                adv_inputs = attacker_train.attack(data, seg)
                data = data.permute(0, 2, 1)
                data=torch.cat((data, adv_inputs))
                seg=torch.cat((seg, seg))
            ###
            #print(data.shape,seg.shape)
            x1,x2,x3,x=model(data)
            seg_pred=head(x1,x2,x3,x)
            #seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            progress_bar(batch_idx, len(train_loader),
                         'Loss: {:.4f}'.format(train_loss / (count +1)))
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
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)
        if epoch % 5 == 0 or epoch > 85:
            ####################
            # Test
            ####################

            test_loss = 0.0
            test_loss_adv= 0.0
            count = 0.0
            model.eval()
            head.eval()
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            test_pred_seg_adv= []
            test_pred_cls_adv=[]
            for data, seg in test_loader:
                data, seg = data.cuda(), seg.cuda()
                adv_inputs = attacker_test.attack(data, seg)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                x1, x2, x3, x = model(data)
                seg_pred = head(x1, x2, x3, x)
                #seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                ### attack count
                x1_, x2_, x3_, x_ = model(adv_inputs)
                seg_pred_adv = head(x1_, x2_, x3_, x_)
                seg_pred_adv = seg_pred_adv.permute(0, 2, 1).contiguous()
                loss_adv = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
                pred_adv = seg_pred_adv.max(dim=2)[1]
                #count_adv += batch_size
                test_loss_adv += loss_adv.item() * batch_size
                #seg_np_adv = seg.cpu().numpy()
                pred_np_adv = pred_adv.detach().cpu().numpy()
                #test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls_adv.append(pred_np_adv.reshape(-1))
                #test_true_seg.append(seg_np)
                test_pred_seg_adv.append(pred_np_adv)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_pred_cls_adv=np.concatenate(test_pred_cls_adv)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            test_acc_adv = metrics.accuracy_score(test_true_cls, test_pred_cls_adv)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_pred_seg_adv = np.concatenate(test_pred_seg_adv, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            test_ious_adv = calculate_sem_IoU(test_pred_seg_adv, test_true_seg)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f, test acc adv: %.6f, test iou_adv: %.6f' % (epoch,
                                                                                                  test_loss*1.0/count,
                                                                                                  test_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(test_ious),
                                                                                                  test_acc_adv,
                                                                                                  np.mean(test_ious_adv))
            io.cprint(outstr)
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                best_acc=test_acc
                torch.save(head.state_dict(), 'outputs/%s/models/head_%s.t7' % (args.exp_name, args.test_area))
    print(best_test_iou,best_acc)

def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1,7):
        visual_file_index = 0
        test_area = str(test_area)
        if os.path.exists("/mnt/ssd/home/junxuan/data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open("/mnt/ssd/home/junxuan/data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
                    visual_file_index = visual_file_index + 1
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            device = torch.device("cuda" if args.cuda else "cpu")
                        
            #Try to load models
            semseg_colors = test_loader.dataset.semseg_colors
            if args.model == 'dgcnn':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")
                
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            model = model.eval()
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1] 
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # visiualization
                visualization(args.visu, args.visu_format, args.test_area, data, seg, pred, visual_file_index, semseg_colors) 
                visual_file_index = visual_file_index + data.shape[0]
            if visual_warning and args.visu != '':
                print('Visualization Failed: You can only choose a room to visualize within the scope of the test area')
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    torch.cuda.set_device(1)
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
    parser.add_argument("--finetune", default=True, type=bool, help='finetune the model')
    parser.add_argument("-w", "--weight", type=str, help="the ckpt path to load")
    parser.add_argument('--budget', type=float, default=0.08,
                        help='FGM attack budget')
    parser.add_argument('--num_iter', type=int, default=15,
                        help='IFGM iterate step')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
