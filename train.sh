#NCCL_DEBUG=INFO  python -m torch.distributed.launch --nproc_per_node=8 BYOL/train.py
torchrun --standalone --nnodes=1 --nproc_per_node=8  BYOL/train_normal.py