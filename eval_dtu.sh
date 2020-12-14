#!/usr/bin/env bash
DTU_TESTING="/home/lab107/data1/linkui/mvs_testing/dtu/"
CKPT_FILE="./checkpoints/edgeflow.ckpt"
OUT_DIR="./outputs_dtu"
python eval_dtu.py --stage=4 --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/dtu_eval.txt --outdir $OUT_DIR --loadckpt $CKPT_FILE $@
