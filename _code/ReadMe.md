### Installation
###### from exists environment
1. conda create -n detreg python=3.8
2. cp -r /home/jovyan/data-vol-1/envs/detreg /home/jovyan/.conda/envs
3. source /opt/conda/bin/activate detreg
4. cd /home/jovyan/data-vol-1/detreg_latest/models/ops && sh make.sh
###### from scratch
1. pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
2. pip install pycocotools tqdm cython scipy opencv-contrib-python seaborn pandas matplotlib

### File structures and formats
1. source data:
    + pretext images: /home/jovyan/nas-dataset/recycling/DF_WuGu_Formal/121K_Pretext/\*.jpg
    + downstream images: /home/jovyan/nas-dataset/recycling/DF_WuGu_Relay_36K_v1_full/\*.jpg
    + downstream labels: /home/jovyan/nas-dataset/recycling/DF_WuGu_Relay_36K_v1_full/\*.txt
2. data preparation: /home/jovyan/data-vol-1/detreg_latest/\_data/
    + pretext/ (see Processes 1)
        + 121k/
            + ilsvrc100/train/Labels.json (not used)
            + ilsvrc100/train/2d/\*.jpg
    + downstream/ (see Processes 4)
        + wuguv1/
            + MSCoco/annotations/instances_train2017.json
            + MSCoco/annotations/instances_val2017.json
            + MSCoco/train2017/\*.jpg
            + MSCoco/val2017/\*.jpg
4. weights and inference outputs: /home/jovyan/data-vol-1/detreg_latest/\_exps/
    + pretext/ (see Processes 2)
        + wugu_121k/
            + checkpoints, log.txt
    + downstream/
        + wuguv1/
            + checkpoints, log.txt (see Processes 5)
            + eval.txt (see Processes 6)
            + output/\*.txt (see Processes 8)
            + pr.npy, ap.jpg, pr.jpg, confusion.jpg, GT_\*\_PD_\*/\*.jpg (see Processes 9)

### \_code1/
###### Packages
1. convert.py: box or labels conversion between coco, voc and yolo
2. detregDownstreamInference.py: save predictions as yolo-liked format
    + cid cx cy w h conf
3. confusion_matrix.py: get confusion matrix with imgPath in blocks
4. visualization.py: save visualized gt and pt images
###### Processes
1. preprocessPretext.ipynb
    + copy all images for imgFolder in imgFolderL -> \_data/pretext/121k/ilsvrc/train/2d/
2. start_pretext.sh / startParallel_pretext.sh
    + input: dataPath=\_data/pretext/121k, outputDIR=\_exps/pretext/wugu_121k, DATA_ROOT_FT=\_data/downstream/labv3 (not used)
    + output at outputDIR: checkpoints, log.txt
3. plotLoss.ipynb
    + load pretext log.txt and plot loss curve
4. preprocessingDownstream.ipynb (call package convert.py for label conversion)
    + copy images from imgFolder and convert from txtFolder (yolo) -> \_data/downstream/wuguv1/MSCoco/
5. start_downstream.sh or startParallel_downstream.sh (arg mode = finetune or finetune_resume)
    + input: DATA_ROOT_FT=\_data/downstream/wuguv1, outputDIR=\_exps/downstream/wuguv1, modelPath=\_exps/pretext/wugu_121k/checkpoint0099.pth
    + output at outputDIR: checkpoints, log.txt
6. start_multi_eval.sh
    + get ap for all weights -> eval.txt
7. plotLoss
    + load downstream log.txt and plot loss curve
    + get best weight from eval.txt
8. start_downstream.sh / start_downstream.sh (arg mode = visualization; call package detregDownstreamInference.py)
    + input: DATA_ROOT_FT=\_data/downstream/wuguv1, outputDIR=\_exps/downstream/wuguv1, modelPath=\_exps/downstream/wuguv1/checkpoint0099.pth
    + output at outputDIR: output/\*.txt
9. result.ipynb (call package convert.py for box conversion; call package confusion_matrix.py; call package visualization.py)
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
