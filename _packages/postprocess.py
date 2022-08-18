class boxAny2Voc:
    def voc(xmin,ymin,xmax,ymax,width=None,height=None):
        return int(xmin),int(ymin), int(xmax), int(ymax)
    def yoloFloat(cx,cy,w,h,width=None,height=None): # width, height only valid
        xmin = int((float(cx)-float(w)/2)*float(width))
        ymin = int((float(cy)-float(h)/2)*float(height))
        xmax = int((float(cx)+float(w)/2)*float(width))
        ymax = int((float(cy)+float(h)/2)*float(height))
        return xmin, ymin, xmax, ymax
    def yoloInt(cx,cy,w,h,width=None,height=None):
        xmin = int(int(cx)-int(w)/2)
        ymin = int(int(cy)-int(h)/2)
        xmax = int(int(cx)+int(w)/2)
        ymax = int(int(cy)+int(h)/2)
        return xmin, ymin, xmax, ymax
    def coco(xmin,ymin,w,h,width=None,height=None):
        return int(xmin), int(ymin), int(xmin)+int(w), int(ymin)+int(h)

def aspectBound(bboxes, boxesType="yoloFloat", threshold=(0,1)):
    assert boxesType=="yoloFloat"
    adopt = []
    for i,(cx,cy,w,h) in enumerate(bboxes):
        if threshold[0] < w/h < threshold[1]:
            adopt.append(i)
    return adopt

def areaBound(bboxes, boxesType="yoloFloat", threshold=(0,1)):
    assert boxesType=="yoloFloat"
    adopt = []
    for i,(cx,cy,w,h) in enumerate(bboxes):
        if threshold[0] < w*h < threshold[1]:
            adopt.append(i)
    return adopt

def IOU(boxA, boxB): # VOC
    (xminA, yminA, xmaxA, ymaxA), (xminB, yminB, xmaxB, ymaxB) = boxA, boxB
    inter = max(0,min(ymaxA,ymaxB)-max(yminA,yminB)) * max(0,min(xmaxA,xmaxB)-max(xminA,xminB))
    areaA = (ymaxA-yminA) * (xmaxA-xminA)
    areaB = (ymaxB-yminB) * (xmaxB-xminB)
    return inter / (areaA+areaB-inter) if (areaA+areaB-inter)!=0 else 0

def NMS(bboxes, boxesType="yoloFloat", threshold=0.3): # bboxes: np.array
    alive, adopt = set(range(len(bboxes))), []
    while len(alive)>=2:
        ma = min(alive)
        adopt.append( ma )
        boxA = getattr(boxAny2Voc,boxesType)(bboxes[ma][0], bboxes[ma][1], bboxes[ma][2], bboxes[ma][3], width=1000, height=1000)
        alive.remove( ma )
        for idx in alive.copy():
            boxB = getattr(boxAny2Voc,boxesType)(bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3], width=1000, height=1000)
            iou  = IOU(boxA,boxB)
            if iou>=threshold:
                alive.remove(idx)
    if len(alive)==1:
        adopt.append(alive.pop())
    return adopt