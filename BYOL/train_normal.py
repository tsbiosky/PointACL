import os
import sys
path = os.getcwd()
sys.path.append(path)

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from BYOL.models.networks_dgcnn import TargetNetwork_DGCNN, OnlineNetwork_DGCNN
from BYOL.models.networks_dgcnn_semseg import TargetNetwork_DGCNN_Semseg, OnlineNetwork_DGCNN_Semseg
from BYOL.models.networks_dgcnn_partseg import TargetNetwork_DGCNN_Partseg, OnlineNetwork_DGCNN_Partseg
#from BYOL.models.networks_votenet import TargetNetwork_VoteNet, OnlineNetwork_VoteNet

from BYOL.models.networks_PointNet import TargetNetwork_PointNet, OnlineNetwork_PointNet
from torchvision import transforms
import math
from collections import OrderedDict
import numpy as np
import BYOL.data.data_utils as d_utils
from BYOL.data.ModelNet40Loader import ModelNet40ClsContrast
from BYOL.data.ShapeNetLoader import PartNormalDatasetContrast, WholeNormalDatasetContrast
from BYOL.data.ScanNetLoader import ScannetWholeSceneContrast, ScannetWholeSceneContrastHeight, ScanNetFrameContrast
from trainer import BYOLTrainer
from BYOL.models.lars_scheduling import LARSWrapper

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


@hydra.main("config/config.yaml")
def main(cfg):
    cfg_dict = hydra_params_to_dotdict(cfg)
    print(cfg_dict)
    model = hydra.utils.instantiate(cfg.task_model, cfg_dict)
    cfg_dict['path']=os.path.join(path, "outputs", cfg.task_model.name)
    online_network = OnlineNetwork_DGCNN(cfg_dict)
    target_network = TargetNetwork_DGCNN(cfg_dict)
    if cfg_dict['resume_ckpt']:
        ckpt = torch.load(cfg_dict['resume_ckpt'])
        state_dict = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key in state_dict:
            if ("online_network." in key or "online_netwrok." in key):
                new_state_dict[key[15:]] = state_dict[key]
        online_network.load_state_dict(new_state_dict)
    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=cfg_dict["optimizer.lr"],
        weight_decay=cfg_dict["optimizer.weight_decay"],
    )
    optimizer = LARSWrapper(optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_dict["epochs"], eta_min=0,
                                                              last_epoch=-1)
    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          **cfg_dict)

    trainer.train()

    with open(os.path.join(path, "outputs", cfg.task_model.name, "cfg.txt"), "w") as file:
        file.write(str(cfg_dict))
        file.write("\n")
    print(cfg_dict)



if __name__ == "__main__":
    main()
