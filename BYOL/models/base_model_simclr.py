import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import math
import numpy as np
import torch
import time
import torch.optim as optim
import BYOL.data.data_utils as d_utils
import diffdist.functional as distops
from scipy.stats import wasserstein_distance
from util.dist_utils import *
from BYOL.data.ModelNet40Loader import ModelNet40ClsContrast
from BYOL.data.ShapeNetLoader import PartNormalDatasetContrast, WholeNormalDatasetContrast
from BYOL.data.ScanNetLoader import ScannetWholeSceneContrast, ScannetWholeSceneContrastHeight, ScanNetFrameContrast
from BYOL.data.data import ModelNet40SSL,ModelNet40_hg
from BYOL.models.lars_scheduling import LARSWrapper

# from BYOL.models.networks import TargetNetwork, OnlineNetwork
from BYOL.models.networks_dgcnn import TargetNetwork_DGCNN, OnlineNetwork_DGCNN
from BYOL.models.networks_dgcnn_semseg import TargetNetwork_DGCNN_Semseg, OnlineNetwork_DGCNN_Semseg
from BYOL.models.networks_dgcnn_partseg import TargetNetwork_DGCNN_Partseg, OnlineNetwork_DGCNN_Partseg
#from BYOL.models.networks_votenet import TargetNetwork_VoteNet, OnlineNetwork_VoteNet
from util.clip_utils import ClipPointsL2
from util.losses import SupConLoss
from util.losses import SinkhornDistance
from BYOL.models.networks_PointNet import TargetNetwork_PointNet, OnlineNetwork_PointNet,SimCLR_PointNet
class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class BasicalModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        assert self.hparams["network"] in ["DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg", "votenet"]
        if self.hparams["network"] == "DGCNN":
            print("Network: DGCNN\n\n\n")
            self.target_network = TargetNetwork_DGCNN(hparams)
            self.online_network = OnlineNetwork_DGCNN(hparams)
        elif self.hparams["network"] == "PointNet":
            print("Network: PointNet\n\n\n")
            self.target_network = SimCLR_PointNet(hparams,pesudo=True)
            self.online_network = SimCLR_PointNet(hparams,pesudo=True)
        elif self.hparams["network"] == "DGCNN-Semseg":
            print("Network: DGCNN for Semseg")
            self.target_network = TargetNetwork_DGCNN_Semseg(hparams)
            self.online_network = OnlineNetwork_DGCNN_Semseg(hparams)
        elif self.hparams["network"] == "DGCNN-Partseg":
            print("Network: DGCNN for Partseg")
            self.target_network = TargetNetwork_DGCNN_Partseg(hparams)
            self.online_network = OnlineNetwork_DGCNN_Partseg(hparams)
        elif self.hparams["network"] == "votenet":
            print("Network: VoteNet for detection")
            self.target_network = TargetNetwork_VoteNet(hparams)
            self.online_network = OnlineNetwork_VoteNet(hparams)
        #self.contrast_criterion = SupConLoss(temperature=0.5)
        self.losslist=[]
        self.losslist12=[]
        self.losslista2=[]
        self.losslista1=[]
        self.budget=0.01
        self.count=-1
        self.num_iter=7
        delta = self.budget
        self.budget = self.budget * \
                      np.sqrt(self.hparams["num_points"] * 3)  # \delta * \sqrt(N * d)
        self.num_iter = int(self.num_iter)
        self.step_size = self.budget / float(self.num_iter)
        self.clip_func = ClipPointsL2(budget=self.budget)
        self.saved_adv = []
        self.saved_aug = []
    def forward(self, pointcloud1, pointcloud2):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        y1, z1 = self.online_network(pointcloud1)
        y2, z2 = self.online_network(pointcloud2)

        return y1,z1,y2,z2

    def regression_loss(self, x, y):
        norm_x = F.normalize(x, dim=1)
        norm_y = F.normalize(y, dim=1)
        loss = 2 - 2 * (norm_x * norm_y).sum() / x.size(0)
        return loss

    def get_current_decay_rate(self, base_tau):
        tau = 1 - (1 - base_tau) * (math.cos(math.pi * self.global_step / (self.epoch_steps * self.hparams["epochs"])) + 1) / 2
        return tau



    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        # use global l2 norm here!
        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm
    def gather(self,z):
        gathered_tensor = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
        gathered_tensor= torch.distributed.all_gather(gathered_tensor, z)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    def NT_xent_multi(self,feat, temperature=0.5):
        feat = feat.permute(2, 1,0)
        view = feat.shape[1]
        feat = torch.cat(torch.unbind(feat, dim=1), dim=0)
        B = feat.shape[0]
        outputs_norm = feat / (feat.norm(dim=1).view(B, 1) + 1e-8)
        similarity_matrix = (1. / temperature) * torch.mm(outputs_norm, outputs_norm.transpose(0, 1).detach())
        N2 = len(similarity_matrix)
        N = int(len(similarity_matrix) / view)
        similarity_matrix_exp = torch.exp(similarity_matrix)
        similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2, N2)).cuda()

        NT_xent_loss = - torch.log(
            similarity_matrix_exp / (torch.sum(similarity_matrix_exp, dim=1).view(N2, 1) + 1e-8) + 1e-8)

        if view==3:
            NT_xent_loss_total = (1. / float(N2)) * torch.sum(
                torch.diag(NT_xent_loss[0:N, N:2 * N]) + torch.diag(NT_xent_loss[N:2 * N, 0:N])
                + torch.diag(NT_xent_loss[0:N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, 0:N])
                + torch.diag(NT_xent_loss[N:2 * N, 2 * N:]) + torch.diag(NT_xent_loss[2 * N:, N:2 * N]))
        elif view==4:
            NT_xent_loss_total = (1. / float(N2)) * torch.sum(
                torch.diag(NT_xent_loss[0:N, N:2 * N]) + torch.diag(NT_xent_loss[N:2 * N, 0:N])
                + torch.diag(NT_xent_loss[0:N, 2 * N:3*N]) + torch.diag(NT_xent_loss[2 * N:3*N, 0:N])
                + torch.diag(NT_xent_loss[0:N, 3 * N:])+ torch.diag(NT_xent_loss[N:2 * N, 3 * N:])
                + torch.diag(NT_xent_loss[N:2 * N, 2 * N:3*N]) + torch.diag(NT_xent_loss[2 * N:3*N, N:2 * N])+torch.diag(NT_xent_loss[2 * N:3*N, 3*N:])
                + torch.diag(NT_xent_loss[3 * N:, 0:N])+torch.diag(NT_xent_loss[3 * N:, N:2*N])+torch.diag(NT_xent_loss[3 * N:, 2*N:3*N]))
        elif view==2:
            NT_xent_loss_total = (1. / float(N2)) * torch.sum(
                torch.diag(NT_xent_loss[0:N, N:]) + torch.diag(NT_xent_loss[N:, 0:N]))
        else:
            raise ValueError('Unknown number of view: {}'.format(view))

        return NT_xent_loss_total

    def NT_xent(self,out_1,out_2,out3=None,temperature=0.5):
        outputs_1=out_1.permute(1, 0)
        outputs_2=out_2.permute(1, 0)
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
    def training_step(self, batch, batch_idx):
        pc_aug1, pc_aug2,hf,label= batch
        if batch_idx==0:
            self.saved_adv=[]
            self.saved_aug=[]
        if self.hparams["network"] in {"DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg"}:
            pc_aug1 = pc_aug1.permute(0, 2, 1)
            pc_aug2 = pc_aug2.permute(0, 2, 1)
            #ori_data = ori_data.permute(0, 2, 1)
        ##perturb
        # B,3,2048
        B,_,K = pc_aug1.shape
        adv_data = pc_aug1.clone()+torch.randn((B, 3, K)).cuda() * 1e-7
        pc_aug1.requires_grad_()
        pc_aug2.requires_grad_()
        adv_data2 = pc_aug2.clone() + torch.randn((B, 3, K)).cuda() * 1e-7
        ce_criterion = nn.CrossEntropyLoss()
        ## attack
        y1, z1, y2, z2 = self.forward(pc_aug1, pc_aug2)
        pred_y1=F.softmax(y1, dim=1)

        with torch.enable_grad():
            for i in range(self.num_iter):
                adv_data.requires_grad_(True)
                adv_data2.requires_grad_(True)
                self.online_network.zero_grad()
                y1, z1, y2, z2 = self.forward(adv_data, pc_aug2)
                #y1, z1, y2, z2 = self.forward(adv_data, adv_data2) # ACL_DS
                logp_hat=F.log_softmax(y1, dim=1)
                adv_loss= F.kl_div(logp_hat, pred_y1, reduction='batchmean') #VAT
                #adv_loss = self.NT_xent(y1, y2) # ROCL&ACL
                #loss = adv_loss + loss_ce * self.ce_weight
                loss=adv_loss
                grad_outputs = None
                grad = torch.autograd.grad(loss, adv_data, grad_outputs=grad_outputs, only_inputs=True, retain_graph=True)[0]
                norm = self.get_norm(grad)
                normalized_grad = grad / (norm[:, None, None] + 1e-9)
                perturbation = self.step_size * normalized_grad
                # add perturbation and clip
                adv_data = adv_data + perturbation
                adv_data = self.clip_func(adv_data, pc_aug1)
                ##ACL
                '''
                grad_outputs = None
                grad = \
                torch.autograd.grad(loss, adv_data2, grad_outputs=grad_outputs, only_inputs=True, retain_graph=True)[0]
                norm = self.get_norm(grad)
                normalized_grad = grad / (norm[:, None, None] + 1e-9)
                perturbation = self.step_size * normalized_grad
                # add perturbation and clip
                adv_data2 = adv_data2 + perturbation
                adv_data2 = self.clip_func(adv_data2, pc_aug2)
                ## ce
                '''


        adv_copy=adv_data.clone()
        adv_copy=adv_copy.detach().cpu().numpy()
        aug_copy=pc_aug1.clone()
        aug_copy =aug_copy.detach().cpu().numpy()
        aug_copy2=pc_aug2.clone()
        aug_copy2=aug_copy2.detach().cpu().numpy()
        self.saved_adv.append(adv_copy)
        self.saved_aug.append(aug_copy)
        #np.savez('./adv_sample_vat.npz', adv=adv_copy.astype('float32'),aug=aug_copy.astype('float32'),aug2=aug_copy2.astype('float32'))
        #print('saved')

        hf=hf.permute(0, 2, 1)
        y1, z1, y2, z2 = self.forward(pc_aug1, pc_aug2)
        y3, z3, y4, z4 = self.forward(hf, adv_data)
        feat=torch.cat([z1.unsqueeze(1), z2.unsqueeze(1),z3.unsqueeze(1),z4.unsqueeze(1)],dim=1)
        #feat = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1),z4.unsqueeze(1)], dim=1)
        loss=self.NT_xent_multi(feat)
        #ACL
        #loss=(self.NT_xent(z1,z2)+self.NT_xent(z3,z4))/2

        #y_ce, z_ce,pred = self.online_network(adv_ce, fc=True)
        #loss_ce = ce_criterion(pred,label)
        #loss=loss+loss_ce*self.ce_weight

        alpha=0
        pred_y1=F.softmax(y1, dim=1)
        logp_hat=F.log_softmax(y4, dim=1)
        kl=F.kl_div(logp_hat, pred_y1, reduction='batchmean')

        #projected
        #pred_y1 = F.softmax(z1, dim=1)
        #logp_hat = F.log_softmax(z4, dim=1)
        #kl = F.kl_div(logp_hat, pred_y1, reduction='batchmean')
        pred_hg=F.softmax(y3, dim=1)
        kl_hg=F.kl_div(logp_hat, pred_hg, reduction='batchmean')
        loss=loss+alpha*kl
        #loss=self.NT_xent(z1,z2)

        log = dict(train_loss=loss)
        return dict(loss=loss, log=log, progress_bar=dict(train_loss=loss))

    def training_epoch_end(self, training_step_outputs):
        '''
        print('record')

        loss=training_step_outputs[-1]['progress_bar']['train_loss']
        lossa2=training_step_outputs[-1]['progress_bar']['adv_loss']
        loss12 = training_step_outputs[-1]['progress_bar']['loss12']
        lossa1 = training_step_outputs[-1]['progress_bar']['lossa1']
        loss_copy = loss.clone().detach().cpu().numpy()
        lossa2_copy = lossa2.clone().detach().cpu().numpy()
        self.losslist.append(loss_copy)
        self.losslist12.append(loss12.detach().cpu().numpy())
        self.losslista1.append(lossa1.detach().cpu().numpy())
        self.losslista2.append(lossa2_copy)
        '''
        saved_aug=np.concatenate(self.saved_aug, axis=0)
        saved_adv=np.concatenate(self.saved_adv,axis=0)
        np.savez('./saved_pc.npz', adv=saved_adv, aug=saved_aug
                 )
        print('saved')
        print(saved_aug.shape)
        self.saved_aug=[]
        self.saved_adv=[]

        return training_step_outputs[-1]

    def validation_step(self, batch, batch_idx):
        pc_aug1, pc_aug2 = batch
        #print(pc_aug1.shape)
        if self.hparams["network"] in {"DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg"}:
            pc_aug1 = pc_aug1.permute(0, 2, 1)
            pc_aug2 = pc_aug2.permute(0, 2, 1)

        y1, z1, y2, z2=self.forward(pc_aug1,pc_aug2)
        loss=self.NT_xent(z1,z2)

        return dict(val_loss=loss)

    def validation_epoch_end(self, outputs):
        reduced_outputs = dict()
        reduced_outputs['val_loss'] = torch.stack([output['val_loss'] for output in outputs]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def configure_optimizers(self):
        if self.hparams["optimizer.type"] == "adam":
            print("Adam optimizer")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            optimizer = LARSWrapper(optimizer)
        elif self.hparams["optimizer.type"] == "adamw":
            print("AdamW optimizer")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"]
            )
        elif self.hparams["optimizer.type"] == "sgd":
            print("SGD optimizer")
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
                momentum=0.9
            )
        else:
            print("LARS optimizer")
            base_optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            optimizer = LARSWrapper(base_optimizer)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["epochs"], eta_min=0,
                                                                  last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=self.hparams["num_points"] * 2, centroid="random"),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
                # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
            ]
        )

        eval_transforms = train_transforms

        train_transforms_scannet_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=self.hparams["num_points"] * 2, centroid="random"),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=np.array([0.0, 0.0, 1.0])),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),

            ]
        )

        train_transforms_scannet_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudRandomInputDropout(p=1),
                # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
            ]
        )

        eval_transforms_scannet_1 = train_transforms_scannet_1
        eval_transforms_scannet_2 = train_transforms_scannet_2

        if self.hparams["dataset"] == "ModelNet40":
            print("Dataset: ModelNet40")
            self.train_dset = ModelNet40_hg(partition='train', num_points=self.hparams["num_points"], normalize=False,transform=train_transforms)

            self.val_dset = ModelNet40SSL(partition='test', num_points=self.hparams["num_points"], normalize=False,transform=eval_transforms)
        elif self.hparams["dataset"] == "ShapeNetPart":
            print("Dataset: ShapeNetPart")
            self.train_dset = PartNormalDatasetContrast(
                self.hparams["num_points"], transforms=train_transforms, split="trainval", normal_channel=True
            )

            self.val_dset = PartNormalDatasetContrast(
                self.hparams["num_points"], transforms=eval_transforms, split="test", normal_channel=True
            )

        elif self.hparams["dataset"] == "ShapeNet":
            print("Dataset: ShapeNet")
            print(self.hparams["num_points"])
            self.train_dset = WholeNormalDatasetContrast(
                self.hparams["num_points"], transforms=train_transforms
            )

            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False, xyz_only=True
            )
        elif self.hparams["dataset"] == "ScanNet":
            print("Dataset: ScanNet")
            self.train_dset = ScannetWholeSceneContrast(
                self.hparams["num_points"], transforms=train_transforms, train=True
            )
            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False, xyz_only=True
            )

        elif self.hparams["dataset"] == "ScanNetFrames":
            print("Dataset: ScanNetFrames")
            self.train_dset = ScanNetFrameContrast(
                self.hparams["num_points"], transforms_1=train_transforms_scannet_1, transforms_2=train_transforms_scannet_2,
                no_height=True, mode=self.hparams["transform_mode"])
            self.val_dset = ScannetWholeSceneContrastHeight(
                self.hparams["num_points"], transforms_1=eval_transforms_scannet_1, transforms_2=eval_transforms_scannet_2, train=False,
                no_height=True)

    def _build_dataloader(self, dset, mode, batch_size=None):
        if batch_size is None:
            batch_size = self.hparams["batch_size"]
        return DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self):
        train_loader = self._build_dataloader(self.train_dset, mode="train")
        self.epoch_steps = len(train_loader)
        return train_loader

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
