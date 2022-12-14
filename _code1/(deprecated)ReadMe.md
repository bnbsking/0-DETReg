Formats
+ ../_data/pretext/bb40k/ilsvrc100/train/2d/*.jpg
+ ../_data/downstream/labv3
    + yolo_train, yolo_val, yolo_test X *.jpg, *.txt, classes.txt
    + classes.txt
    + voc_train, voc_val, voc_test X *.xml
    + MSCoco
        + train2017, val2017: *.jpg
        + annotations/instances_train2017.json, annotations/instances_val2017.json 
+ ../_exps/downstream/test3k_coco/viz_txt || yolo-analog: cid, conf, xmin(int), ymin, xmax, ymax 

Files
1. preprocessing.ipynb
    + pretext part: collect folders with images and copy to speficied folder
    + downstream part: 
        + ../_packages/preprocess.py:
            + (optional) check (intersection,shape,class,bounds,aspectRatio&size), plot (bar,example), auto-balance, show specific or examples after well done
            + copy images and annotation to yolo_train/val/test
            + convert to voc_train/val/test (annotations only)
            + convert to MSCoco (both images and annotations)
        + ../_packages/convert.py: Yolo <-> VOC <-> Coco (annotations only)
        + ../_packages/visualization.py
2. training:
    + start_pretext.sh: ep, bs, data path, resume
    + startParallel_pretext.sh: above and cores
    + start_downstream.sh: (ep,bs,data path,resume) X (finetune,finetune_resume,evaluation,visualization)
    + startParallel_downstream.sh: above and cores
3. plotLoss.ipynb
4. testing:
    + start_multi_eval.sh: data path, weight path -> get best weight
    + start_downstream.sh: visualization mode
    + ../_packages/detregDownstreamInference.py
        + load prefix and classList from Coco
        + save output as yolo-analog format and visualization 
5. result.ipynb
    + ../_packages/result.py
        + read gt in Voc and pd in yolo-analog format as list[np.array]
        + get P/R/PR/AP/confusion/blockImgs by ../_packages/confusion_matrix.py
