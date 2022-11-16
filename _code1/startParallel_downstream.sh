#!/bin/bash

mode_arr=(finetune finetune_resume evalutation visualization)
MODE_IDX=0
OUTPUT_DIR="_exps/downstream/test3k_recycle_parallel"
LR_DROP=40
EPOCHS=50
#FINETUNE_WEIGHT="_exps/pretext/official/checkpoint_coco.pth"
FINETUNE_WEIGHT="_exps/pretext/test3k_parallel/checkpoint0049.pth"
#FINETUNE_WEIGHT="_exps/downstream/test3k_coco_parallel/checkpoint0049.pth"
DATA_ROOT_FT="_data/downstream/test3k"

if [ $MODE_IDX == 0 ]; then
    #python -u main.py --output_dir $OUTPUT_DIR --dataset coco --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --epochs $EPOCHS --lr_drop $LR_DROP
    GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_full_coco.sh --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --epochs $EPOCHS --lr_drop $LR_DROP 

elif [ $MODE_IDX == 1 ]; then
    GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_full_coco.sh --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --epochs $EPOCHS --lr_drop $LR_DROP

elif [ $MODE_IDX == 2 ]; then
    GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_full_coco.sh --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --eval

else
    GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_full_coco.sh --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 1 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --viz

fi
