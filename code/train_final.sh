#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node 4 train_final.py --model_name "samh_unet_final" --batch_size 4 --max_epochs 50
