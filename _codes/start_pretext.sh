#!/bin/bash

OUTPUT_DIR="_exps/pretext/xavier_bb40k"
LR_DROP=999
EPOCHS=200
DATA_ROOT="_data/pretext/bb40k"
DATA_ROOT_FT="_data/downstream/labv3"

python -u main.py --output_dir $OUTPUT_DIR --dataset imagenet100 --strategy topk --load_backbone swav --max_prop 30 --object_embedding_loss --lr_backbone 0 --epochs $EPOCHS --batch_size 24 --num_workers 32 --lr_drop $LR_DROP --data_root $DATA_ROOT --data_root_ft $DATA_ROOT_FT --resume _exps/pretext/xavier_bb40k/checkpoint0169.pth