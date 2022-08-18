import json, glob, re, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import confusion_matrix as CM

class Result:
    """
    gtPath  = "/home/jovyan/data-vol-1/detreg_latest/_data/downstream/c7_0620/voc_val"
    pdPath  = "/home/jovyan/data-vol-1/detreg_latest/_exps/downstream/coco_40k_ep200_c70620/viz_txt"
    imgPath = "/home/jovyan/data-vol-1/detreg_latest/_exps/downstream/coco_40k_ep200_c70620/viz_jpg"
    classesTxtPath = "/home/jovyan/data-vol-1/detreg_latest/_data/downstream/c7_0620/yolo_val/classes.txt"
    classL = ['PLC','NPLC','PAC','NPAC','Tetra','HDPE','PET']
    classNumL = [146, 54, 2115, 3306, 248, 11, 23]
    obj = result.Result(gtPath, pdPath, imgPath, classesTxtPath, classL, classNumL)
    obj.plotAP()
    obj.plotPR()
    obj.plotConfusion(strategy="precision")
    obj.getBlockImgs(2,0)
    """
    def __init__(self, gtPath, pdPath, imgPath, classesTxtPath, classL, classNumL, savePath=None):
        self.gtPathL = sorted(glob.glob(f"{gtPath}/*.xml")) # groundtruth
        self.pdPathL = sorted(glob.glob(f"{pdPath}/*.txt")) # detections (cid,conf,xmin,ymin,xmax,ymax)
        self.imgPath = imgPath
        assert len(self.gtPathL)==len(self.pdPathL)==len(glob.glob(f"{imgPath}/*.jpg"))
        self.classesTxtL = list(filter(lambda c:c,map(lambda c:c.replace('\n',''),open(classesTxtPath,"r").readlines())))
        self.classL = classL
        self.classNumL = classNumL
        assert len(self.classesTxtL)==len(classL)==len(classNumL)
        self.savePath= savePath if savePath else "/".join(pdPath.split("/")[:-1]+['results'])
        os.makedirs(self.savePath, exist_ok=True)
        self._getLabelDetection()
        self._getPR() # self.PR=[{"precision":[.]*101,"recall":[.]*101} for i in range(n) ]
        self._getRefineRP()
        self._getAPs()
    
    def _getLabelDetection(self):
        self.labels, self.detections = [], []
        for gtFilePath,pdFilePath in zip(self.gtPathL,self.pdPathL):
            xml = open(gtFilePath,"r").read()
            nameL = list(map( lambda name:self.classesTxtL.index(name), re.findall("<name>(.*)</name>",xml) ))
            xminL = list(map( lambda xmin:int(xmin), re.findall("<xmin>(.*)</xmin>",xml) ))
            yminL = list(map( lambda ymin:int(ymin), re.findall("<ymin>(.*)</ymin>",xml) ))
            xmaxL = list(map( lambda xmax:int(xmax), re.findall("<xmax>(.*)</xmax>",xml) ))
            ymaxL = list(map( lambda ymax:int(ymax), re.findall("<ymax>(.*)</ymax>",xml) ))
            self.labels.append( np.array(list(zip(nameL,xminL,yminL,xmaxL,ymaxL))) ) # cid,xmin,ymin,xmax,ymax
            detections = [ line.replace("\n","").split(" ") for line in open(pdFilePath,"r").readlines()[:100] ]
            self.detections.append( np.array(list(map(lambda L:[int(L[2]),int(L[3]),int(L[4]),int(L[5]),float(L[1]),int(L[0])],detections))) ) # xmin,ymin,xmax,ymax,conf,cid
    
    def _xgetLabelDetection(self):
        self.labels, self.detections = [], []
        for i,(gtFilePath,pdFilePath) in enumerate(zip(self.gtPathL,self.pdPathL)):
            labels     = [ line.replace("\n","").split(" ") for line in open(gtFilePath,"r").readlines() ]
            self.labels.append( np.array(list(map(lambda L:[int(L[0]),int(L[1]),int(L[2]),int(L[3]),int(L[4])],labels))) ) # cid,xmin,ymin,xmax,ymax
            detections = [ line.replace("\n","").split(" ") for line in open(pdFilePath,"r").readlines()[:100] ]
            self.detections.append( np.array(list(map(lambda L:[int(L[2]),int(L[3]),int(L[4]),int(L[5]),float(L[1]),int(L[0])],detections))) ) # xmin,ymin,xmax,ymax,conf,cid
    
    def _getPR(self):
        prPath = f"{self.savePath}/pr.npy"
        if glob.glob(prPath):
            print("load pr.npy from cache")
            self.PR = np.load(prPath, allow_pickle=True)
        else:
            n = len(self.classL)
            self.PR = [ {"precision":[0.]*101, "recall":[0.]*101} for i in range(n) ]
            for i in range(101):
                print(f"\rthreshold={round(i*0.01,2)}{'0' if i%10==0 else ''}", end="")
                M = np.zeros( (n+1,n+1) ) # col:gt, row:pd
                for j,(labels,detections) in enumerate(zip(self.labels,self.detections)):
                    cm = CM.ConfusionMatrix(n, CONF_THRESHOLD=i*0.01, IOU_THRESHOLD=0.5)
                    cm.process_batch(detections,labels)
                    M += cm.return_matrix()
                #
                rowSum, colSum = M.sum(axis=1), M.sum(axis=0)
                for j in range(n):
                    self.PR[j]["precision"][i] = M[j][j]/rowSum[j] if rowSum[j] else 0
                    self.PR[j]["recall"][i]    = M[j][j]/colSum[j] if colSum[j] else 0
            np.save(prPath, self.PR)
            print()
    
    def _getRefineRP(self): # sorted by recall, and enhance precision by next element reversely
        n = len(self.classL)
        for i in range(n):
            R, P = self.PR[i]["recall"][:], self.PR[i]["precision"][:]
            Z = sorted(zip(R,P))
            R, P = zip(*Z)
            R, P = list(R), list(P)
            for j in range(1,101):
                P[-1-j] = max(P[-1-j], P[-j])
            self.PR[i]["refineRecall"], self.PR[i]["refinePrecision"] = R, P

    def _getAPs(self):
        self.apL, self.mAP, self.wmAP, n = [], 0, 0, len(self.classL)
        for i in range(n):
            ap = 0
            for j in range(100):
                ap += self.PR[i]["refinePrecision"][j]*(self.PR[i]["refineRecall"][j+1]-self.PR[i]["refineRecall"][j])
            self.apL.append( round(ap,3) )
            self.mAP  += ap/n
            self.wmAP += ap*self.classNumL[i]/sum(self.classNumL)
        self.mAP  = round(float(self.mAP),3)
        self.wmAP = round(float(self.wmAP),3)
        print(f"apL={self.apL}, mAP={self.mAP}, wmAP={self.wmAP}")
    
    def plotAP(self, showMAP=False, showAPS=False):
        R = list(range(len(self.classL)))
        plt.figure()
        ax = plt.subplot(1,1,1)
        ax.set_title(f"{'mAP='+str(self.mAP)+', ' if showAPS else ''}wmAP={self.wmAP}", fontsize=16)
        ax.bar(self.classL, self.apL)
        if showAPS:
            for i in range(len(self.classL)):
                ax.text(R[i], self.apL[i], self.apL[i], ha="center", va="bottom", fontsize=16)
        plt.savefig(f"{self.savePath}/ap.jpg")
        plt.show()
                    
    def plotPR(self):
        n = len(self.classL)
        plt.figure(figsize=(6*n,12))
        for i in range(n):
            plt.subplot(2,n,1+i)
            plt.scatter( self.PR[i]["refineRecall"], self.PR[i]["refinePrecision"] )
            plt.plot( self.PR[i]["refineRecall"], self.PR[i]["refinePrecision"] )
            plt.xlim(-0.05,1.05)
            plt.ylim(-0.05,1.05)
            plt.grid('on')
            plt.title(f"class-{i}", fontsize=16)
            plt.xlabel("recall", fontsize=16)
            plt.ylabel("precision", fontsize=16)

            plt.subplot(2,n,1+i+n)
            plt.plot( self.PR[i]["precision"])
            plt.plot( self.PR[i]["recall"])
            plt.plot( [(self.PR[i]["precision"][j]+self.PR[i]["recall"][j])/2 for j in range(101)] )
            plt.xlim(-5,105)
            plt.ylim(-0.05,1.05)
            plt.grid('on')
            plt.title(f"class-{i}", fontsize=16)
            plt.xlabel("threshold", fontsize=16)
            plt.legend(labels=["precision","recall","fvalue"], fontsize=12)
        plt.savefig(f"{self.savePath}/pr.jpg")
        plt.show()
    
    def _getBestThreshold(self):
        n = len(self.classL)
        wScore = [0]*101
        for i in range(101):
            for j in range(n):                
                p, r = self.PR[j]["precision"][i], self.PR[j]["recall"][i]
                if self.strategy=="fvalue":
                    score = 2*p*r/(p+r) if p+r else 0
                elif self.strategy=="precision":
                    score = p if r>=0.5 else 0
                else:
                    raise
                wScore[i] += score*self.classNumL[j]/sum(self.classNumL)
        bestScore, self.bestThreshold = max(zip(wScore,[round(0.01*i,2) for i in range(101)]))
        print(f"bestScore={round(bestScore,2)}, best_threshold={self.bestThreshold}")
    
    def plotConfusion(self, strategy="fvalue"):
        self.strategy = strategy
        self._getBestThreshold()
        n = len(self.classL)
        M = np.zeros( (n+1,n+1) ) # col:gt, row:pd
        self.accumFileL = [ [[] for j in range(n+1)] for i in range(n+1) ] # (n+1,n+1) each grid is path list
        for j,(gtPath,labels,detections) in enumerate(zip(self.gtPathL,self.labels,self.detections)):
            cm = CM.ConfusionMatrix(n, CONF_THRESHOLD=self.bestThreshold, IOU_THRESHOLD=0.5, gtFile=gtPath.split("/")[-1], accumFileL=self.accumFileL)
            cm.process_batch(detections,labels)
            M += cm.return_matrix()
            self.accumFileL = cm.getAccumFileL()
        #
        axis0sum = M.sum(axis=0)
        N = M.copy()
        for i in range(len(N)):
            if axis0sum[i] != 0:
                N[:,i] /= axis0sum[i]
        axis1sum = M.sum(axis=1)
        P = M.copy()
        for i in range(len(P)):
            if axis1sum[i] != 0:
                P[i,:] /= axis1sum[i]
        #
        plt.figure(figsize=(15,5))
        # fig1 - number
        fig = plt.subplot(1,3,1)
        plt.title(f"Confusion Matrix - Number (conf={self.bestThreshold})", fontsize=12)
        plt.xlabel("GT", fontsize=12)
        plt.ylabel("PD", fontsize=12)
        fig.set_xticks(np.arange(n+1)) # values
        fig.set_xticklabels(self.classL+['BG']) # labels
        fig.set_yticks(np.arange(n+1)) # values
        fig.set_yticklabels(self.classL+['BG']) # labels
        plt.imshow(P, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(n+1):
            for j in range(n+1):
                plt.text(j, i, int(M[i][j]), ha="center", va="center", color="black" if P[i][j]<0.9 else "white", fontsize=12)
        # fig2 - precision
        fig = plt.subplot(1,3,2)
        plt.title(f"Confusion Matrix - Row norm (Precision)", fontsize=12)
        plt.xlabel("GT", fontsize=12)
        plt.ylabel("PD", fontsize=12)
        fig.set_xticks(np.arange(n+1)) # values
        fig.set_xticklabels(self.classL+['BG']) # labels
        fig.set_yticks(np.arange(n+1)) # values
        fig.set_yticklabels(self.classL+['BG']) # labels
        plt.imshow(P, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(n+1):
            for j in range(n+1):
                plt.text(j, i, round(P[i][j],2), ha="center", va="center", color="black" if P[i][j]<0.9 else "white", fontsize=12)
        # fig3 - recall
        fig = plt.subplot(1,3,3)
        plt.title(f"Confusion Matrix - Col norm (Recall)", fontsize=12)
        plt.xlabel("GT", fontsize=12)
        plt.ylabel("PD", fontsize=12)
        fig.set_xticks(np.arange(n+1)) # values
        fig.set_xticklabels(self.classL+['BG']) # labels
        fig.set_yticks(np.arange(n+1)) # values
        fig.set_yticklabels(self.classL+['BG']) # labels
        plt.imshow(N, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(n+1):
            for j in range(n+1):
                plt.text(j, i, round(N[i][j],2), ha="center", va="center", color="black" if N[i][j]<0.9 else "white", fontsize=12)
        #plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.cm.Blues))
        plt.savefig(f"{self.savePath}/confusion.jpg")
        plt.show()
        json.dump(self.accumFileL, open(f"{self.savePath}/confusionFiles.json","w"))
        
    def getBlockImgs(self, row, col): # PD, GT
        classL = self.classL + ['BG']
        folder = f"{self.savePath}/GT_{classL[col]}_PD_{classL[row]}"
        os.makedirs(folder, exist_ok=True)
        for path in map(lambda file:f"{self.imgPath}/{file.split('.')[0]}.jpg",self.accumFileL[row][col]):
            os.system(f"cp {path} {folder}")
        print(f"len(glob.glob(folder+'/*.jpg'))={len(glob.glob(folder+'/*.jpg'))}")