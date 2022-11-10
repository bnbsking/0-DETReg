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
