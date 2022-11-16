# calling relation: main.py -> engine.py -> detregDownstreamInference.py
import json, os
import numpy as np
import cv2
import torch

D, classList, gtImgFolder = {}, [], ""

def main(outputs, image_id, savePath, data_root_ft):
    global D, classList, gtImgFolder
    if not D:
        D = json.load( open(f"{data_root_ft}/MSCoco/annotations/instances_val2017.json","r") )
        classList = [ d['name']for d in D["categories"] ]
        gtImgFolder = f"{data_root_ft}/MSCoco/val2017"
        os.makedirs(f"{savePath}/output", exist_ok=True)
    
    prefix = D['images'][image_id]['file_name'].split('.')[0]
    bboxes = outputs['pred_boxes'][0]        # (300,(cx,cy,w,h)) float
    classScores = outputs['pred_logits'][0]  # (300,classes+1) float
    classScores = torch.sigmoid(classScores) # (300,classes+1) float
    topConfIdx  = classScores.max(axis=1).values.sort(descending=True).indices # (300,) int
    bboxes = bboxes[topConfIdx]                                                # (300,(cx,cy,w,h)) float
    scores, classes = classScores[topConfIdx].max(axis=1)                      # (300,) float, (300,) int
    classes= classes # index start from 0 or 1
            
    with open(f"{savePath}/output/{prefix}.txt", "w") as f:
        height, width, _ = cv2.imread(f"{gtImgFolder}/{prefix}.jpg").shape
        for cid, score, (cx,cy,w,h) in zip(classes,scores,bboxes):
            if score>0.01:
                f.write(f"{cid} {round(float(cx),5)} {round(float(cy),5)} {round(float(w),5)} {round(float(h),5)} {round(float(score),3)}\n")