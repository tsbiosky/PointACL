# PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack



This is the official code implementation for the paper "PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack" 


### Downstream Tasks

+ [x] Shape Classification
+ [x] Semantic Segmentation

## Datasets

Please download the used dataset with the following links:
+ ModelNet40: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

## Pre-training

### BYOL framework

Please run the following command:

```
python BYOL/train.py
```

You need to edit the config file `BYOL/config/config.yaml` to switch different backbone architectures (currently including `BYOL-pointnet-cls, BYOL-dgcnn-cls, BYOL-dgcnn-semseg, BYOL-votenet-detection`).

### Pre-trained Models

You can find the checkpoints of the pre-training and downstream tasks in our [Google Drive](https://drive.google.com/drive/folders/1uip_uZtyoVdTbwUM4QUZmEcXROpV2-8n?usp=sharing).

## Linear Evaluation

For PointNet or DGCNN classification backbones, you may evaluate the learnt representation with linear SVM classifier by running the following command:

For PointNet:

```
python BYOL/evaluate_pointnet.py -w /path/to/your/pre-trained/checkpoints
```

For DGCNN:

```
python BYOL/evaluate_dgcnn.py -w /path/to/your/pre-trained/checkpoints
```

## Downstream Tasks

### Checkpoints Transformation

You can transform the pre-trained checkpoints to different downstream tasks by running:

For VoteNet:

```
python BYOL/transform_ckpt_votenet.py --input_path /path/to/your/pre-trained/checkpoints --output_path /path/to/the/transformed/checkpoints
```

For other backbones:

```
python BYOL/transform_ckpt.py --input_path /path/to/your/pre-trained/checkpoints --output_path /path/to/the/transformed/checkpoints
```

### Fine-tuning and Evaluation for Downstream Tasks

For the fine-tuning and evaluation of downstream tasks, please refer to other corresponding repos. We sincerely thank all these authors for their nice work!

+ Classification: [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)
+ Semantic Segmentation: [AnTao97/*dgcnn*.pytorch](https://github.com/AnTao97/dgcnn.pytorch)
+ Indoor Object Detection: [facebookresearch/*votenet*](https://github.com/facebookresearch/votenet)

## Citation

If you found our paper or code useful for your research, please cite the following paper:

```
@article{huang2021spatio,
  title={Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds},
  author={Huang, Siyuan and Xie, Yichen and Zhu, Song-Chun and Zhu, Yixin},
  journal={arXiv preprint arXiv:2109.00179},
  year={2021}
}
```
