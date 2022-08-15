import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
import numpy as np
import BYOL.data.data_utils as d_utils
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from tqdm import tqdm
from BYOL.data.ModelNet40Loader import ModelNet40ClsContrast
from BYOL.data.ShapeNetLoader import PartNormalDatasetContrast, WholeNormalDatasetContrast
from BYOL.data.ScanNetLoader import ScannetWholeSceneContrast, ScannetWholeSceneContrastHeight, ScanNetFrameContrast

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

class BYOLTrainer:
    def __init__(self, online_network, target_network, optimizer, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.max_epochs = params['epochs']
        self.writer = SummaryWriter()
        #self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        local_rank = int(os.environ["LOCAL_RANK"])
        self.hparams = params
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True
        self.update_module(self.target_network, self.online_network, decay_rate=0)
        self.tau = self.hparams["decay_rate"]
    def update_module(self, target_module, online_module, decay_rate):
        online_dict = online_module.state_dict()
        target_dict = target_module.state_dict()
        for key in target_dict:
            target_dict[key] = decay_rate * target_dict[key] + (1 - decay_rate) * online_dict[key]
        target_module.load_state_dict(target_dict)
    def get_current_decay_rate(self, base_tau):
        tau = 1 - (1 - base_tau) * (math.cos(math.pi * self.global_step / (self.epoch_steps * self.hparams["epochs"])) + 1) / 2
        return tau
    def regression_loss(self, x, y):
        norm_x = F.normalize(x, dim=1)
        norm_y = F.normalize(y, dim=1)
        loss = 2 - 2 * (norm_x * norm_y).sum() / x.size(0)
        return loss
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
        y1_online, z1_online, qz1_online = self.online_network(pointcloud1)
        y2_online, z2_online, qz2_online = self.online_network(pointcloud2)

        with torch.no_grad():
            y1_target, z1_target = self.target_network(pointcloud1)
            y2_target, z2_target = self.target_network(pointcloud2)

        return y1_online, qz1_online, y2_online, qz2_online, y1_target, z1_target, y2_target, z2_target
    def train(self):
        self.prepare_data()
        train_loader = self.train_dataloader()

        niter = 0

        for epoch_counter in range(self.max_epochs):

            for (pc_aug1, pc_aug2, ori) in tqdm(train_loader):
                pc_aug1 = pc_aug1.to(self.device)
                pc_aug2 = pc_aug2.to(self.device)
                if self.hparams["network"] in {"DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg"}:
                    pc_aug1 = pc_aug1.permute(0, 2, 1)
                    pc_aug2 = pc_aug2.permute(0, 2, 1)
                ##drop
                print('here')
                y1_online, qz1_online, y2_online, qz2_online, y1_target, z1_target, y2_target, z2_target \
                    = self.forward(pc_aug1, pc_aug2)
                loss = self.regression_loss(qz1_online, z2_target)
                loss += self.regression_loss(qz2_online, z1_target)
                self.writer.add_scalar('loss', loss, global_step=niter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.tau = self.get_current_decay_rate(self.hparams["decay_rate"])
                self.update_module(self.target_network, self.online_network, decay_rate=self.tau)
                niter += 1
            print("End of epoch {}".format(epoch_counter))

            model_checkpoints_folder=self.hparams['path']
            # save checkpoints
            self.save_model(os.path.join(model_checkpoints_folder, 'model_epoch:'+str(epoch_counter)+'.pth'))

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
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
            self.train_dset = ModelNet40ClsContrast(
                self.hparams["num_points"],adv_data=self.hparams["adv_data"], transforms=train_transforms, train=True
            )

            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False
            )
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
