#!/bin/bash

OUTPUT_DIR="_exps/downstream/xavier_bb40k_ep199"
DATA_ROOT_FT="_data/downstream/labv3"

for FINETUNE_WEIGHT in $(ls $OUTPUT_DIR | grep checkpoint012); do
    FINETUNE_WEIGHT=$OUTPUT_DIR/$FINETUNE_WEIGHT
    echo ---$FINETUNE_WEIGHT---
    echo ---$FINETUNE_WEIGHT--- >> $OUTPUT_DIR/eval.txt
    python -u main.py --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --eval >> $OUTPUT_DIR/eval.txt
done