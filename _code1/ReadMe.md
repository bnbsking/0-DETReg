### File structures and formats
1. .../\[\*.img,\*.txt\]
2. \_data/
    + pretext/
        + 121k/
            + ilsvrc100/train/Labels.json
            + ilsvrc100/train/2d/\*.jpg
    + downstream/
        + wuguv1/
            + MSCoco/annotations/instances_train2017.json
            + MSCoco/annotations/instances_val2017.json
            + MSCoco/train2017/\*.jpg
            + MSCoco/val2017/\*.jpg
3. \_exps/
    + pretext/
        + wugu_121k/
            + checkpoints, log.txt
    + downstream/
        + wuguv1/
            + checkpoints, log.txt
            + eval.txt
            + output/\*.txt
            + pr.npy, ap.jpg, pr.jpg, confusion.jpg
            + GT_\*\_PD_\*/\*.jpg

### \_code1/
###### packages
1. convert.py: box or labels conversion between coco, voc and yolo
2. detregDownstreamInference.py: save predictions as yolo-liked format
    + cid cx cy w h conf
3. confusion_matrix.py: get confusion matrix with imgPath in blocks
4. visualization.py: save visualized gt and pt images
###### process
1. preprocessPretext.ipynb
    + copy all images for imgFolder in imgFolderL -> \_data/pretext/121k/ilsvrc/train/2d/
2. start_pretext.sh / startParallel_pretext.sh
    + input: dataPath=\_data/pretext/121k, outputDIR=\_exps/pretext/wugu_121k, DATA_ROOT_FT=\_data/downstream/labv3 (make no sense)
    + output at outputDIR: checkpoints, log.txt
3. plotLoss.ipynb
    + load pretext log.txt and plot loss curve
4. preprocessingDownstream.ipynb (call convert.py for label conversion)
    + copy images from imgFolder and convert from txtFolder (yolo) -> \_data/downstream/wuguv1/MSCoco/
5. start_downstream.sh / startParallel_downstream.sh (mode=finetune/finetune_resume)
    + input: DATA_ROOT_FT=\_data/downstream/wuguv1, outputDIR=\_exps/downstream/wuguv1, modelPath=\_exps/pretext/wugu_121k/checkpoint0099.pth
    + output at outputDIR: checkpoints, log.txt
6. start_multi_eval.sh
    + get ap for all weights -> eval.txt
7. plotLoss
    + load downstream log.txt and plot loss curve
    + get best weight from eval.txt
8. start_downstream.sh / start_downstream.sh (mode=visualization; call detregDownstreamInference.py)
    + input: DATA_ROOT_FT=\_data/downstream/wuguv1, outputDIR=\_exps/downstream/wuguv1, modelPath=\_exps/downstream/wuguv1/checkpoint0099.pth
    + output at outputDIR: output/\*.txt
9. result.ipynb (call convert.py for box conversion; call confusion_matrix.py; call visualization.py)
    + specify imgFolder and load labels .../\*.txt as well as predictions output/\*.txt
    + output pr.npy, ap.jpg, pr.jpg, confusion.jpg and GT_\*\_PD_\*/\*.jpg
        
### Modified
+ main.py
    + line 325-326: dump config as json
    + line 212: add data_root_ft as input argument
+ util/default_args.py
    + line 129: add data_root_ft as an argument
+ models/__init__.py]
    + line 32-33: setting num_classes **(Modify if class changed)***
+ engine.py
    + line 200,218: add data_root_ft as input argument
    + line 215-219: call ddi module
+ \_code1/confusion_matrix.py
    + compare with https://github.com/kaanakan/object_detection_confusion_matrix/blob/master/confusion_matrix.py
+ models/segmentation.py
    + line 215: imbalance training
