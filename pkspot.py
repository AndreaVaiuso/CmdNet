import cv2
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

from tensorflow.python.eager.context import num_gpus
import classifier
from utilities import printErr

CLASSIFIER = "NETWORK/malexcnr.keras"
IMG_FORMAT = ".jpg"
TOL = 20

class Patch:
    def  __init__(self,name,imgArray,imsource,x1,x2,y1,y2):
       self.name = name
       self.imgArray = imgArray
       self.confidence = 1
       self.imsource = imsource
       self.x1 = x1
       self.x2 = x2
       self.y1 = y1
       self.y2 = y2

def getFName(name):
    n = name.split(".")
    text = ""
    i = 0
    for i in range(len(n)):
        if i == len(n)-2:
            text += n[i]
            return text
        else:
            text += n[i] + "."

def orderArrays(list1,path1,list2,path2):
    nlist1 = [""]*len(list1)
    nlist2 = [""]*len(list2)
    i = 0
    for name1 in path1:
        name1 = getFName(name1)
        j = 0
        for name2 in path2:
            name2 = getFName(name2)
            if name1 == name2:
                nlist1[i] = list1[i]
                nlist2[i] = list2[j]
            j += 1
        i += 1
    return nlist1, nlist2

def addToPatches(patches,ptc):
    for patchList in patches:
        if patchList[0].name == ptc.name:
            patchList.append(ptc)
            ln = len(patchList)
            n_conf = 1/ln
            for patch in patchList:
                patch.confidence = n_conf
            return patches
    patches.append([ptc])
    return patches

##For result compute weighted average!    

def weightedMean(val,w):
    i = 0
    num = 0
    den = 0
    for i in range(len(val)):
        num += val[i] * w[i]
        den += w[i]
    return num/den

def setConfidence(patches,camname,parkname,factor):
    for patchlist in patches:
        confs = []
        alpha = 0
        
        if patchlist[0].name != parkname:
            continue
        for patch in patchlist:
            if getFName(patch.imsource) != camname:
                confs.append(patch.confidence)
                continue
            alpha = patch.confidence
        if len(confs) == 0:
            return patches
        nfac = (1 - (factor * alpha)) / sum(confs)
        for patch in patchlist:
            if getFName(patch.imsource) != camname:
                patch.confidence = patch.confidence*nfac
            else: patch.confidence = patch.confidence*factor
    return patches

def main():
    #RULES! The name of XML file must match the name of image file
    xmlpath = sys.argv[1]
    imgpath = sys.argv[2]
    xmlRoots = []
    imgArrays = []
    xmldirs = []
    imgdirs = []
    for file in os.listdir(xmlpath):
        if file.endswith(".xml"):
            xmldirs.append(file)
    for file in os.listdir(imgpath):
        if file.endswith(IMG_FORMAT):
            imgdirs.append(file)        
    ln = len(xmldirs)
    if ln != len(imgdirs):
        printErr("Missing frames...")
        return
    for xmlfile in xmldirs:
        xmlRoots.append(ET.parse(xmlpath+"/"+xmlfile).getroot())
    for imgfile in imgdirs:
        imgArrays.append(cv2.imread(imgpath+"/"+imgfile))

    xmlRoots, imgArrays = orderArrays(xmlRoots,xmldirs,imgArrays,imgdirs)

    patches = []

    for i in range(ln):
        for child in xmlRoots[i]:
            patchname = ""
            x1 = 0
            x2 = 0
            y1 = 0
            y2 = 0
            for patch in child:
                if patch.tag == "NAME":
                    patchname = patch.text
                if patch.tag == "X1":
                    x1 = int(patch.text)
                if patch.tag == "X2":
                    x2 = int(patch.text)
                if patch.tag == "Y1":
                    y1 = int(patch.text)
                if patch.tag == "Y2":
                    y2 = int(patch.text)
            ptc = Patch(patchname,imgArrays[i][y1:y2,x1:x2],imgdirs[i],x1,x2,y1,y2)
            patches = addToPatches(patches,ptc)

    patches = setConfidence(patches,"CAM3","Parking 8", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 9", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 10", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 11", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 12", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 13", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 14", 0.1)
    patches = setConfidence(patches,"CAM3","Parking 15", 0.1)


    classifier.loadModel(CLASSIFIER)

    predResult = []

    for plist in patches:
        tempResult = []
        confs = []
        for patch in plist:
            prediction, timeForClassific = classifier.classify(patch.imgArray)
            pred = (prediction[0][0] - 0.5) * 2
            pred = pred * 100
            result = ""
            if pred > 0: result = "free"
            else: result = "occupied"
            classifResult = "{:.2f}% ".format(abs(pred)) + result
            tempResult.append(pred)
            confs.append(patch.confidence)
            print(patch.name,"-",patch.imsource,":",classifResult,"[",patch.confidence,"]")
        predResult.append(weightedMean(tempResult,confs))
    print()
    i = 0
    for res in predResult:
        result = ""
        if res > 0: result = "free"
        else: result = "occupied"
        classifResult = "{:.2f}% ".format(abs(res)) + result
        print(patches[i][0].name,classifResult,"[",str(len(patches[i])),"cameras ]")
        i += 1
    showOverlays(imgdirs,patches,predResult)

def showOverlays(imgdirs,patches,preds):
    red_color = (0, 0, 255, 255)
    green_color = (0, 255, 0, 255)
    white_color = (255, 255, 255, 255)
    
    imgs = []
    imgpath = sys.argv[2]
    for imgfile in imgdirs:
        rgb_data = cv2.imread(imgpath+"/"+imgfile)
        rgba = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
        imgs.append(rgba)
    
    i = 0
    for i in range(len(imgs)):
        img = imgs[i]
        x,y,ch = img.shape
        overlay = np.zeros((x,y,4))
        ptcdex = 0
        for ptc in patches:
            for patch in ptc:
                if patch.imsource == imgdirs[i]:
                    result = preds[ptcdex]
                    if result>TOL: color = green_color
                    elif result<-TOL: color = red_color
                    else: color = white_color
                    overlay = cv2.rectangle(overlay,(patch.x1,patch.y1),(patch.x2,patch.y2),color,3) 
            ptcdex +=1
        res = imgs[i]
        cnd = overlay[:, :, 3] > 0
        res[cnd] = overlay[cnd]
        cv2.imshow("Overlay " + imgdirs[i],res)
        cv2.waitKey()
        i += 1

if __name__ == "__main__":
    main()

    
