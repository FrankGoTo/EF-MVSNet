#!/usr/bin/env bash
DTU_TESTING="/home/lab107/data1/linkui/tankandtemples/intermediate/"
CKPT_FILE="./checkpoints/edgeflow.ckpt"
OUT_DIR="./outputs_tanks"
python eval_tank.py --stage=1 --dataset=tanks --batch_size=1 --testpath=$DTU_TESTING --testlist lists/tanks/tank_eval.txt --outdir $OUT_DIR --loadckpt $CKPT_FILE $@
