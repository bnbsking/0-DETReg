import glob, os, json, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from convert import yolo2voc, voc2coco
from visualization import show

class PreparePretext:
    """
    # available for both creating data or adding new data to current folder
    obj = pre.PreparePretext(imgFolderL=["/home/jovyan/data-vol-2/recycling/Lab/extra_data/label/"], destName="40k")
    obj.checkShape()
    obj.createPretextFolder()
    obj.copy()
    """
    def __init__(self, imgFolderL=[], destName="example"):
        self.pathL = self.getPathL(imgFolderL)
        self.destPath = os.path.dirname(os.path.abspath(__file__)) + f"/../_data/pretext/{destName}"
    
    def getPathL(self, imgFolderL):
        pathL = []
        for imgFolder in imgFolderL:
            paths = glob.glob(f"{imgFolder}/*.jpg")
            print(imgFolder, len(paths))
            pathL += paths
        print(f"total length = {len(pathL)}")
        return pathL
    
    def checkShape(self):
        self.shapeD = {}
        for i,path in enumerate(self.pathL):
            print(f"\r{i+1}/{len(self.pathL)}", end="")
            img = cv2.imread(path)
            shape = img.shape if hasattr(img,"shape") else None
            self.shapeD[shape] = self.shapeD[shape]+[path] if shape in self.shapeD else [path]
        print("\n", {key:len(self.shapeD[key]) for key in self.shapeD} )
    
    def removeInvalidShape(self, shapeL=[]):
        for shape in shapeL:
            for path in self.D[shape]:
                self.pathL.remove(path)
    
    def createPretextFolder(self):
        os.makedirs(self.destPath, exist_ok=True)
        os.makedirs(f"{self.destPath}/ilsvrc100", exist_ok=True)
        os.makedirs(f"{self.destPath}/ilsvrc100/train", exist_ok=True)
        os.makedirs(f"{self.destPath}/ilsvrc100/train/2d", exist_ok=True)
        json.dump({"2d":"2d"}, open(f"{self.destPath}/ilsvrc100/train/Labels.json","w"))
    
    def copy(self):
        for i,path in enumerate(self.pathL):
            print(f"\r{i+1}/{len(self.pathL)}", end="")
            os.system(f"cp {path} {self.destPath}/ilsvrc100/train/2d")
        print(f"\nlen(os.listdir({self.destPath}/ilsvrc100/train/2d))={ len(os.listdir(self.destPath+'/ilsvrc100/train/2d')) }")

        
        
class PrepareDownstream: # change yolo float
    """
    obj = pre.PrepareDownstream(\
        trainImgFolder="/home/jovyan/data-vol-2/recycling/Lab/train_v3/recycle_data_5/train/2d",
        valImgFolder="/home/jovyan/data-vol-2/recycling/Lab/train_v3/recycle_data_5/valid/valid_2d",
        testImgFolder="/home/jovyan/data-vol-2/recycling/Lab/test_v2_5classes",
        blackS=blackS, classL=['PLC','NPLC','PAC','NPAC','Tetra'], destName="labv3")
    obj.checkShape(["test"])
    obj.checkClassesBounds(["train","val","test"])
    obj.checkAspectArea()
    obj.autoBalanceTrain()
    obj.plotClassNum()
    obj.checkAspectArea()
    obj.show()
    # obj.showSpecific(imgPath,antPath)
    obj.removeInvalidSet()
    obj.copy()
    obj.yolo2voc()
    obj.voc2coco()
    obj.yolo2ap()
    """
    def __init__(self, trainImgFolder=None, trainAntFolder=None, valImgFolder=None, valAntFolder=None, testImgFolder=None, testAntFolder=None,\
            blackS=set(), classesTxtPath=None, classL=None, destName="example"):
        self.trainPathL = self._getInterPrefix(trainImgFolder, trainAntFolder if trainAntFolder else trainImgFolder, blackS, "train")
        self.valPathL   = self._getInterPrefix(valImgFolder, valAntFolder if valAntFolder else valImgFolder, blackS, "val")
        self.testPathL  = self._getInterPrefix(testImgFolder, testAntFolder if testAntFolder else testImgFolder, blackS, "test")
        self.classesTxtPath = classesTxtPath if classesTxtPath else f"{trainImgFolder}/classes.txt"
        self.classL   = classL
        self.destPath = os.path.dirname(os.path.abspath(__file__)) + f"/../_data/downstream/{destName}"
        
    def _getInterPrefix(self, imgFolder, antFolder, blackS, name="train"):
        imgPrefix  = map(lambda path:path.split('/')[-1].split('.')[0], glob.glob(f"{imgFolder}/*.jpg"))
        antPrefix  = map(lambda path:path.split('/')[-1].split('.')[0], glob.glob(f"{antFolder}/*.txt"))
        imgPrefixS = set(filter(lambda prefix:prefix not in blackS,imgPrefix))
        antPrefixS = set(filter(lambda prefix:prefix not in blackS,antPrefix))
        prefixS = imgPrefixS.intersection(antPrefixS)
        pathL = sorted(list(map(lambda prefix:(f"{imgFolder}/{prefix}.jpg",f"{antFolder}/{prefix}.txt"),prefixS)))
        print( f"{name}, len(imgPrefixS)={len(imgPrefixS)}, len(antPrefixS)={len(antPrefixS)}, len(self.{name}PathL)={len(pathL)}" )
        return pathL
    
    def checkShape(self, groups=["train","val","test"]):
        for group in groups:
            pathL = getattr(self,f"{group}PathL")
            shapeD = {}
            for i,(imgPath,_) in enumerate(pathL):
                print(f"\r{i+1}/{len(pathL)}", end="")
                img = cv2.imread(imgPath)
                shape = img.shape if hasattr(img,"shape") else None
                shapeD[shape] = shapeD[shape]+[imgPath] if shape in shapeD else [imgPath]
            setattr(self, f"{group}ShapeD", shapeD)
            print( f"\nself.{group}ShapeD is created, summary:", sorted([(key,len(shapeD[key])) for key in shapeD]) )
            
    def checkClassesBounds(self, groups=["train","val","test"]):
        for group in groups:
            pathL = getattr(self,f"{group}PathL")
            classD, outBoundS = {}, set()
            for i,(_,antPath) in enumerate(pathL):
                print(f"\r{i+1}/{len(pathL)}", end="")
                for line in open(antPath,"r").readlines():
                    cid, cx, cy, w, h = line.replace("\n","").split(" ")
                    classD[cid] = classD[cid]+[antPath] if cid in classD else [antPath]
                    if not 0<=float(cx)<=1 or not 0<=float(cy)<=1 or not 0<=float(w)<=1 or not 0<=float(h)<=1:
                        outBoundS.add(antPath)
            setattr(self, f"{group}ClassD", classD)
            setattr(self, f"{group}OutBoundS", outBoundS)
            print(f"\nself.{group}ClassD is created, summary:", sorted([(cid,len(classD[cid])) for cid in classD]) )
            print(f"self.{group}OutBoundS is created, summary:", { path.split("/")[-1] for path in outBoundS } )
    
    def autoBalanceTrain(self):
        trainAvgN = int( sum([len(self.trainClassD[key]) for key in sorted(self.trainClassD.keys())]) / len(self.classL) )
        print(f"Auto balance each class to {trainAvgN} data as far as possible")
        newTrainPathL = []
        newTrainNL = [0]*len(self.classL)
        random.shuffle(self.trainPathL)

        abandonIdx = set()
        for i in range( len(self.trainPathL)*10 ):
            if len(newTrainPathL) >= trainAvgN*len(self.classL):
                break
            elif i%len(self.trainPathL) in abandonIdx:
                continue
            trainPath, antPath = self.trainPathL[i%len(self.trainPathL)]
            lineL = open(antPath, "r").readlines()
            cidL = list(map(lambda line:int(line.split(" ")[0]), lineL))
            newTrainNLTemp = newTrainNL[:]
            adopt = True
            for cid in cidL:
                if newTrainNLTemp[cid]+1 > trainAvgN:
                    adopt = False
                    abandonIdx.add( i%len(self.trainPathL) )
                    break
                else:
                    newTrainNLTemp[cid]+=1
            if adopt:
                newTrainPathL.append( (trainPath,antPath) )
                newTrainNL = newTrainNLTemp[:]
        self.trainPathL = newTrainPathL[:]
        self.checkClassesBounds(['train'])
        print(f"len(self.trainPathL)={len(self.trainPathL)}, len(set(self.trainPathL))={len(set(self.trainPathL))}")
    
    def plotClassNum(self):
        trainNL = [ len(self.trainClassD[key]) for key in sorted(self.trainClassD.keys()) ] if hasattr(self,"trainClassD") else [0]*len(self.classL)
        trainRL = list(map(lambda x:round(x/sum(trainNL),2),trainNL))
        valNL   = [ len(self.valClassD[key]) for key in sorted(self.valClassD.keys()) ] if hasattr(self,"valClassD") else [0]*len(self.classL)
        valRL   = list(map(lambda x:round(x/sum(valNL),2),valNL))
        testNL   = [ len(self.testClassD[key]) for key in sorted(self.testClassD.keys()) ] if hasattr(self,"testClassD") else [0]*len(self.classL)
        testRL   = list(map(lambda x:round(x/sum(testNL),2),testNL))
        R, width = np.arange(len(self.classL)), 0.3
        #
        ax = plt.subplot(1,1,1)
        plt.title(f"train:{sum(trainNL)}, val:{sum(valNL)}, test:{sum(testNL)}", fontsize=16)
        plt.bar(R-width, trainRL, width)
        plt.bar(R, valRL, width)
        plt.bar(R+width, testRL, width)
        for i in range(len(self.classL)):
            ax.text(R[i]-width, trainRL[i], trainNL[i], ha="center", va="bottom", fontsize=12)
            ax.text(R[i], valRL[i], valNL[i], ha="center", va="bottom", fontsize=12)
            ax.text(R[i]+width, testRL[i], testNL[i], ha="center", va="bottom", fontsize=12)
        plt.ylabel("data number", fontsize=16)
        plt.xticks(R, self.classL)
        plt.legend(labels=["train","val","test"])
        plt.grid('on')
        os.makedirs(self.destPath, exist_ok=True)
        plt.savefig(f"{self.destPath}/classes.jpg")
        plt.show()
        json.dump({"trainNL":trainNL, "valNL":valNL, "testNL":testNL}, open(f"{self.destPath}/classes.json","w"))
    
    def checkAspectArea(self, groups=["train","val","test"]):
        for group in groups:
            pathL = getattr(self,f"{group}PathL")
            aspectD, areaD = {}, {}
            for i,(_,antPath) in enumerate(pathL):
                print(f"\r{i+1}/{len(pathL)}", end="")
                for line in open(antPath,"r").readlines():
                    cid, cx, cy, w, h = line.replace("\n","").split(" ")
                    aspectD[antPath] = round(float(w)/float(h),4)
                    areaD[antPath] = round(float(w)*float(h),8)
            setattr(self, f"{group}AspectD", aspectD)
            setattr(self, f"{group}AreaD", areaD)
            aspectL = np.array(list(aspectD.values()))
            areaL   = np.array(list(areaD.values()))
            print(f"\nself.{group}AspectD is created, max={aspectL.max()}, min={aspectL.min()}, mean={round(aspectL.mean(),4)}, std={round(aspectL.std(),4)}")
            print(  f"self.{group}AreaD is created, max={areaL.max()}, min={areaL.min()}, mean={round(areaL.mean(),8)}, std={round(areaL.std(),8)}" )
            
    def removeInvalidSet(self, prefixS=set(), groups=["train","val","test"]):
        for group in groups:
            pathL = getattr(self,f"{group}PathL")
            newPathL = []
            for i,(imgPath,antPath) in enumerate(pathL):
                if antPath.split("/")[-1].split(".")[0] not in prefixS:
                    newPathL.append( (imgPath,antPath) )
            setattr(self, f"{group}PathL", newPathL)
            print(f"len(self.{group}PathL)={len(newPathL)}")
    
    def copy(self, groups=["train","val","test"]):
        for group in groups:
            destPath = f"{self.destPath}/yolo_{group}"
            os.makedirs(destPath, exist_ok=True)
            pathL = getattr(self,f"{group}PathL")
            repeat = {}
            for i,(imgPath,antPath) in enumerate(pathL):
                print(f"\r{i+1}/{len(pathL)}", end="")
                if (imgPath,antPath) not in repeat:
                    os.system(f"cp {imgPath} {destPath} && cp {antPath} {destPath}")
                    repeat[(imgPath,antPath)]=1
                else:
                    imgName = imgPath.split('/')[-1].replace('.jpg',f"_{repeat[(imgPath,antPath)]}.jpg")
                    antName = antPath.split('/')[-1].replace('.txt',f"_{repeat[(imgPath,antPath)]}.txt")
                    os.system(f"cp {imgPath} {destPath}/{imgName} && cp {antPath} {destPath}/{antName}")
                    repeat[(imgPath,antPath)]+=1
            os.system(f"cp {self.classesTxtPath} {destPath}")
            print(f"\nlen(glob.glob({destPath}/*.txt))={len(glob.glob(destPath+'/*.txt'))}")
            print(  f"len(glob.glob({destPath}/*.jpg))={len(glob.glob(destPath+'/*.jpg'))}")
            
    def yolo2voc(self, groups=["train","val","test"]):
        for group in groups:
            srcPath = f"{self.destPath}/yolo_{group}"
            destPath = f"{self.destPath}/voc_{group}"
            os.makedirs(destPath, exist_ok=True)
            yolo2voc(srcPath, destPath)
            print(f"len(glob.glob({destPath}/*.xml))={len(glob.glob(destPath+'/*.xml'))}")
            
    def voc2coco(self, groups=["train","val","test"]):
        os.makedirs(f"{self.destPath}/MSCoco", exist_ok=True)
        os.makedirs(f"{self.destPath}/MSCoco/annotations", exist_ok=True)
        for group in groups:
            # annotations
            removeNewLineMap = map(lambda s:s.replace('\n',''), open(self.classesTxtPath,"r").readlines())
            classList = list(filter(lambda s:s,removeNewLineMap))
            voc2coco(f"{self.destPath}/voc_{group}", f"{self.destPath}/MSCoco/annotations/instances_{group}2017.json", classList); print()
            # images
            os.makedirs(f"{self.destPath}/MSCoco/{group}2017", exist_ok=True)
            imgPathL = glob.glob(f"{self.destPath}/yolo_{group}/*.jpg")
            for i,imgPath in enumerate(imgPathL):
                print(f"\r{i+1}/{len(imgPathL)}", end="")
                os.system(f"cp {imgPath} {self.destPath}/MSCoco/{group}2017")
            print(f"\nlen(glob.glob({self.destPath}/MSCoco/{group}2017/*)={len(glob.glob(self.destPath+'/MSCoco/'+group+'2017/*'))}")
        print(f"os.listdir({self.destPath}/MSCoco/annotations)={os.listdir(self.destPath+'/MSCoco/annotations')}")
        
    def show(self, trainN=5, valN=5, testN=5):
        classList = list(map(lambda line:line.replace("\n",""), open(self.classesTxtPath,"r").readlines()))
        _ = os.makedirs(f"{self.destPath}/viz_train", exist_ok=True) if trainN else None
        _ = os.makedirs(f"{self.destPath}/viz_val", exist_ok=True) if valN else None
        _ = os.makedirs(f"{self.destPath}/viz_test", exist_ok=True) if testN else None
        for _ in range(trainN):
            r = random.randint(0,len(self.trainPathL)-1)
            show( self.trainPathL[r][0], self.trainPathL[r][1], classList=classList, savePath=f"{self.destPath}/viz_train", valueRatios=(1.5,1.5) )
        for _ in range(valN):
            r = random.randint(0,len(self.valPathL)-1)
            show( self.valPathL[r][0], self.valPathL[r][1], classList=classList, savePath=f"{self.destPath}/viz_val", valueRatios=(1.5,1.5) )
        for _ in range(testN):
            R = random.randint(0,len(self.testPathL)-1)
            show( self.testPathL[r][0], self.testPathL[r][1], classList=classList, savePath=f"{self.destPath}/viz_test", valueRatios=(1.5,1.5) )
            
    def showSpecific(self, imgPath, antPath):
        classList = list(map(lambda line:line.replace("\n",""), open(self.classesTxtPath,"r").readlines()))
        show( imgPath, antPath, classList=classList, savePath=self.destPath, valueRatios=(1.5,1.5) )
        
#     def _yolo2ap(self, group="test"):
#         os.makedirs(f"{self.destPath}/map_{group}", exist_ok=True) # annotation only
#         antPathL = list(filter(lambda s:"classes.txt" not in s, glob.glob(f"{self.destPath}/yolo_{group}/*.txt")))
#         for i,antPath in enumerate( antPathL ):
#             print(f"\r{i+1}/{len(antPathL)}", end="")
#             height, width, _ = cv2.imread(antPath.replace('.txt','.jpg')).shape
#             with open(antPath.replace(f"yolo_{group}",f"map_{group}"),'w') as f:
#                 for line in open(antPath,"r").readlines():
#                     cid, cx, cy, w, h = line.replace('\n','').split(" ")
#                     xmin = int((float(cx)-float(w)/2)*width)
#                     ymin = int((float(cy)-float(h)/2)*height)
#                     xmax = int((float(cx)+float(w)/2)*width)
#                     ymax = int((float(cy)+float(h)/2)*height)
#                     f.write(f"{cid} {xmin} {ymin} {xmax} {ymax}\n")
#         print(f"\nlen(os.listdir({self.destPath}/map_{group}))={len(os.listdir(self.destPath+'/map_'+group))}")