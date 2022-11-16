#!/bin/bash

OUTPUT_DIR="_exps/downstream/example"
DATA_ROOT_FT="_data/downstream/example"

for FINETUNE_WEIGHT in $(ls $OUTPUT_DIR | grep checkpoint0); do
    FINETUNE_WEIGHT=$OUTPUT_DIR/$FINETUNE_WEIGHT
    echo ---$FINETUNE_WEIGHT---
    echo ---$FINETUNE_WEIGHT--- >> $OUTPUT_DIR/eval.txt
    python -u main.py --output_dir $OUTPUT_DIR --dataset coco --pretrain $FINETUNE_WEIGHT --batch_size 4 --num_workers 8 --data_root_ft $DATA_ROOT_FT --resume $FINETUNE_WEIGHT --eval >> $OUTPUT_DIR/eval.txt
done
