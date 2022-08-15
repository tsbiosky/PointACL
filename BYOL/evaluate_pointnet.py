import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import SVC

from BYOL.models import PointNet
from BYOL.data import ModelNet40Cls
from BYOL.data.data import ModelNet40

def load_model(weight_path, model):
    state_dict = model.state_dict()

    ckpt = torch.load(weight_path, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if "online_network." + key in pretrained_dict:
            state_dict[key] = pretrained_dict["online_network."+key]
            print(key)

    model.load_state_dict(state_dict, strict=True)
    return model


def fetch_represent(loader, model):
    representations = None
    labels = None
    batch_num = len(loader)
    for batch_id, data in enumerate(loader):
        if (batch_id + 1) % 100 == 0:
            print("%d/%d" % (batch_id+1, batch_num))
        pc, label = data
        pc = pc.to(device)
        pc = pc.permute(0, 2, 1)
        with torch.no_grad():
            representation = model(pc).cpu().numpy()
        if representations is None:
            representations = representation
            labels = label
        else:
            representations = np.concatenate([representations, representation], 0)
            labels = np.concatenate([labels, label], 0)
    return representations, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", type=str, help="the ckpt path to load")
    parser.add_argument("--xyz_only", default=1, type=str, help="whether to only use xyz-coordinate for evaluation")
    parser.add_argument("--num_points", default=2048, type=int)
    parser.add_argument("--k", default=40, type=int, help="choose gpu")
    parser.add_argument("--dropout", default=0.5, type=float, help="choose gpu")
    parser.add_argument("--emb_dims", default=1024, type=int, help="dimension of hidden embedding")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size of dataloader")
    parser.add_argument("--gpu_num", default=0, type=int, help="choose gpu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hparam = load_hparam(args.model_yaml)
    hparam = {"model.use_xyz": True, "emb_dims": args.emb_dims, "dropout":args.dropout, "num_points": args.num_points, "k": args.k}

    model = PointNet(hparam)
    model = load_model(args.weight, model)
    model.to(device)
    model.eval()

    train_dataset = ModelNet40(args.num_points, partition='train')
    val_dataset = ModelNet40(args.num_points, partition='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print("Fetch Train Data Representation")
    train_represent, train_label = fetch_represent(train_loader, model)

    print("Fetch Val Data Representation")
    val_represent, val_label = fetch_represent(val_loader, model)

    svc = SVC(kernel="linear", verbose=False)
    svc.fit(train_represent, train_label)

    score = svc.score(val_represent, val_label)
    print("Val Accuracy:", score)



