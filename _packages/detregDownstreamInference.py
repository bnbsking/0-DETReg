# calling relation: main.py -> engine.py -> detregDownstreamInference.py
import cv2, sys, json, os
packagePath = os.path.dirname(os.path.abspath(__file__))
sys.path += [] if packagePath in sys.path else [packagePath]
import postprocess as pp
import visualization as vz
import torch

D, classList, gtImgFolder, gtAntFolder = {}, [], "", ""

def main(outputs, image_id, savePath, data_root_ft, aspectBound=(0.1,6), areaBound=(0.001,0.75), NMS=0.5, savetxt=True, confThreshold=0.30, savejpg=True):
    global D, classList, gtImgFolder, gtAntFolder
    if not D:
        D = json.load( open(f"{data_root_ft}/MSCoco/annotations/instances_val2017.json","r") )
        classList = [ d['name']for d in D["categories"] ]
        gtImgFolder = f"{data_root_ft}/yolo_val"
        gtAntFolder = f"{data_root_ft}/yolo_val"
        os.makedirs(f"{savePath}/viz_txt" if savetxt else ".", exist_ok=True)
        os.makedirs(f"{savePath}/viz_jpg" if savejpg else ".", exist_ok=True)
    
    prefix = D['images'][image_id]['file_name'].split('.')[0]
    bboxes = outputs['pred_boxes'][0]                       # (300,(cx,cy,w,h)) float
    if False: # direct softmax
        classScores = outputs['pred_logits'][0].softmax(-1) # (300,classes+1) float
        # !!! 0: for other team # 1: for our team 
        topConfIdx = classScores[:,1:].max(axis=1).values.sort(descending=True).indices
        # get max conf of each query then get their ranking index # (300,) int
        bboxes = bboxes[topConfIdx]                         # (300,(cx,cy,w,h)) float
        scores, classes = classScores[topConfIdx][:,1:].max(axis=1) # (300,) float, (300,) int
        # removeNums = {} # removeNums["crossCategoryNMS"] = len(bboxes)-len(adopt) # removeNums["confidenceThreshold"] = len(bboxes)-len(adopt)
    else:
        # unify background last (300,classes+BG) float, then remove background queries (N,classes+BG) float
        if 1: # backgroundLast
            classScores = outputs['pred_logits'][0]
        else:
            classScores = outputs['pred_logits'][0][:,list(range(1,outputs['pred_logits'][0].shape[1]))+[0]]
        foregrounds = classScores.max(axis=1).values>0 # (F,) bool
        bboxes      = bboxes[foregrounds]
        classScores = classScores[foregrounds]
        # sort by confidence
        if len(bboxes):
            maxsco = classScores.max(axis=1).values                         # (N,) float
            scoidx = maxsco.sort(descending=True).indices                   # (N,) int
            bboxes = bboxes[scoidx]                                         # (N,4) float
            scores = classScores[scoidx].softmax(axis=1).max(axis=1).values # (N,) float
            classes= classScores[scoidx].argmax(axis=1)                     # (N,) int
            #print(prefix, bboxes, scores, classes)#;raise
        else:
            scores, classes = torch.Tensor([]), torch.Tensor([])
    
    if False and aspectBound:
        adopt = pp.aspectBound(bboxes.cpu().numpy(), boxesType="yoloFloat", threshold=aspectBound)
        bboxes, scores, classes = bboxes[adopt], scores[adopt], classes[adopt]
        
    if False and areaBound:
        adopt = pp.areaBound(bboxes.cpu().numpy(), boxesType="yoloFloat", threshold=areaBound)
        bboxes, scores, classes = bboxes[adopt], scores[adopt], classes[adopt]
    
    if False and NMS: # cross category
        adopt = pp.NMS(bboxes.cpu().numpy(), boxesType="yoloFloat", threshold=NMS)
        bboxes, scores, classes = bboxes[adopt], scores[adopt], classes[adopt]
            
    if savetxt:
        with open(f"{savePath}/viz_txt/{prefix}.txt", "w") as f:
            height, width, _ = cv2.imread(f"{gtImgFolder}/{prefix}.jpg").shape
            for cid, score, (cx,cy,w,h) in zip(classes,scores,bboxes):
                score= round(float(score),4) 
                xmin = int((float(cx)-float(w)/2)*width)
                ymin = int((float(cy)-float(h)/2)*height)
                xmax = int((float(cx)+float(w)/2)*width)
                ymax = int((float(cy)+float(h)/2)*height)
                f.write(f"{cid} {score} {xmin} {ymin} {xmax} {ymax}\n")
            
    if savejpg:
        if confThreshold:
            adopt = list(filter(lambda i:scores[i]>=confThreshold, range(len(bboxes))))
            bboxes, scores, classes = bboxes[adopt], scores[adopt], classes[adopt]
        vz.show(f"{gtImgFolder}/{prefix}.jpg", f"{gtAntFolder}/{prefix}.txt", "yoloFloat", bboxes, classes, scores, classList, f"{savePath}/viz_jpg", (1.5,1.5))