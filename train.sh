#!/usr/bin/env bash
MVS_TRAINING="/home/lab107/data1/linkui/mvs_training/dtu/"
# CKPT="./checkpoints/edgeflow.ckpt"
python train.py --dataset=dtu_yao --batch_size=1 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/d192 $@
