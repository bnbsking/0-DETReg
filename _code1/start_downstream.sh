#!/bin/bash

mode_arr=(finetune finetune_resume evalutation visualization)
MODE_IDX=3
OUTPUT_DIR="_exps/downstream/example"
LR_DROP=40
EPOCHS=10
#FINETUNE_WEIGHT="_exps/pretext/official/checkpoint_coco.pth"
FINETUNE_WEIGHT="_exps/downstream/example/checkpoint0009.pth"
DATA_ROOT_FT="_data/downstream/example"

if [ $MODE_IDX == 0 ]; then
    #python -u main.py --output_dir $OUTPUT_DIR --dataset coco --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --epochs $EPOCHS --lr_drop $LR_DROP
    python -u main.py --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --epochs $EPOCHS --lr_drop $LR_DROP 

elif [ $MODE_IDX == 1 ]; then
    python -u main.py --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --epochs $EPOCHS --lr_drop $LR_DROP

elif [ $MODE_IDX == 2 ]; then
    python -u main.py --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --eval

else
    python -u main.py --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 1 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --viz

fi
