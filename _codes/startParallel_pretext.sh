#!/bin/bash

OUTPUT_DIR="_exps/pretext/test3k_parallel_120b"
LR_DROP=45
EPOCHS=50
DATA_ROOT="_data/pretext/test3k"
DATA_ROOT_FT="_data/downstream/labv3"

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_top30_in100.sh --output_dir $OUTPUT_DIR --batch_size 24 --num_workers 8 --epochs $EPOCHS --lr_drop $LR_DROP --data_root $DATA_ROOT --data_root_ft $DATA_ROOT_FT
#--resume _exps/pretext/xavier_bb40k/checkpoint0169.pth