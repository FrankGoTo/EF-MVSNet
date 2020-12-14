#!/usr/bin/env bash
TESTPATH="/home/lab107/data1/linkui/mvs_training/dtu/"
TESTLIST="lists/dtu/val.txt"
CKPT_FILE="checkpoints/edgeflow.ckpt"
python train.py --batch_size=1 --mode="test" --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:2}
