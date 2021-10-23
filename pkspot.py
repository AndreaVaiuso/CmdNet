import cv2
import os
import sys
import xml.etree.ElementTree as ET

from tensorflow.python.eager.context import num_gpus
import classifier
from utilities import printErr

CLASSIFIER = "NETWORK/malexcnr.keras"
IMG_FORMAT = ".jpg"

class Patch:
    def  __init__(self,name,imgArray):
       self.name = name
       self.imgArray = imgArray
       self.confidence = 1

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

    i = 0
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
            ptc = Patch(patchname,imgArrays[i][y1:y2,x1:x2])
            patches = addToPatches(patches,ptc)
    i += 1

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
        predResult.append(weightedMean(tempResult,confs))
    
    print(predResult)


if __name__ == "__main__":
    main()

    
