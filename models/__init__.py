# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from .backbone import build_backbone
from .deformable_detr import DeformableDETR, SetCriterion as DefSetCriterion, PostProcess as DefPostProcess
from .detr import DETR, SetCriterion as DETRSetCriterion, PostProcess as DETRPostProcess
from .def_matcher import build_matcher as build_def_matcher
from .detr_matcher import build_matcher as build_detr_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer

from .transformer import build_transformer


def build_model(args):
    if args.dataset_file == 'coco':
        num_classes = 90
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
    elif args.dataset_file == 'airbus':
        num_classes = 1
    else:
        num_classes = 20
    num_classes += 1
    device = torch.device(args.device)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef} # 2:5:2
    if args.masks: # False
        weight_dict["loss_mask"] = args.mask_loss_coef # 1
        weight_dict["loss_dice"] = args.dice_loss_coef # 1
    # TODO this is a hack
    if args.aux_loss: # True
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1): # 6-1
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})

        # only in def detr impl.
        if args.model == 'deformable_detr': # True
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict) # loss_ce_/0~4/enc (2), loss_bbox_/0~4/enc (5), loss_giou_/0~4/enc (2) # 7*3=21

    losses = ['labels', 'boxes', 'cardinality']
    if args.object_embedding_loss: # True
        losses.append('object_embedding')
        weight_dict['loss_object_embedding'] = args.object_embedding_coef # 1

    if args.masks: # False
        losses += ["masks"]

    backbone = build_backbone(args)

    if args.model == 'deformable_detr':
        transformer = build_deforamble_transformer(args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes, # 
            num_queries=args.num_queries, # 300
            num_feature_levels=args.num_feature_levels, # 4
            aux_loss=args.aux_loss, # True
            with_box_refine=args.with_box_refine, # False
            two_stage=args.two_stage, # False
            object_embedding_loss=args.object_embedding_loss, # True
            obj_embedding_head=args.obj_embedding_head # intermediate
        )
        matcher = build_def_matcher(args)
        criterion = DefSetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
        postprocessors = {'bbox': DefPostProcess()}

    elif args.model == 'detr':
        transformer = build_transformer(args)
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head
        )
        matcher = build_detr_matcher(args)
        criterion = DETRSetCriterion(num_classes, matcher, weight_dict, args.eos_coef,
                                     losses, object_embedding_loss=args.object_embedding_loss)
        postprocessors = {'bbox': DETRPostProcess()}
    else:
        raise ValueError("Wrong model.")

    criterion.to(device)

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
