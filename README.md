# PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack(ICASSP 2023)



This is the official code implementation for the paper "PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack" 
![image](https://github.com/tsbiosky/PointACL/blob/master/overview%203.png)

### Downstream Tasks

+ [x] Shape Classification
+ [x] Semantic Segmentation

## Datasets

Please download the used dataset with the following links:
+ ModelNet40: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

## Pre-training

### SimCLR framework

Please run the following command:

```
python BYOL/train.py
```

You need to edit the config file `BYOL/config/config.yaml` to switch different backbone architectures


##  Evaluation on Downstream Tasks

Coming soon
```


