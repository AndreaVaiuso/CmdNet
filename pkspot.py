import cv2
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import keras
import json
import datetime
import time

from utilities import secToTime, darker, weightedMean, getFName
from tensorflow.python.eager.context import num_gpus
import classifier
from utilities import printErr
from threading import Thread

CLASSIFIER = "NETWORK/malexcnr.keras"
IMG_FORMAT = ".jpg"
TOL = 15

FONT_COLOR = (255,255,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
LINE_TYPE = 1

prediction_results = {}

class Patch:
    """Patch class contains all necessary parameters to identify a single parking spot inside a parking lot image
    :param name: ID of the parking spot
    :param imgArray: RGB pixel matrix that represent the parking spot
    :param imsource: File name of the parking lot image source that contains that parking spot image
    :param x1: Upper left pixel coordinate of parking spot patch cointanied in the parking lot image source
    :param x2: Upper right pixel coordinate of parking spot patch cointanied in the parking lot image source
    :param y1: Lower left pixel coordinate of parking spot patch cointanied in the parking lot image source
    :param y2: Lower right pixel coordinate of parking spot patch cointanied in the parking lot image source
    """
    def  __init__(self,name:str,imgArray:list,imsource:str,x1:int,x2:int,y1:int,y2:int):
        """Constructor method
        """
        self.name = name
        self.imgArray = imgArray
        self.confidence = 1
        self.imsource = imsource
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

def updatePatches(imgdirs:list,imgpath:str,patches:list):
    """Update all image data for each patch in patches list
    :param imgdirs: List containing all image file names of all frames to be update
    :param imgpath: Path that contains all image files
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image:
    :return: Updated patches list
    :rtype: list
    """
    imgArrays = {}
    for imgfile in imgdirs:
        imgArrays[imgfile] = cv2.imread(imgpath+"/"+imgfile)
    for patchList in patches:
        for patch in patchList:
            if imgArrays[patch.imsource] is None: continue
            patch.imgArray = imgArrays[patch.imsource][patch.y1:patch.y2,patch.x1:patch.x2]
    return patches

def addToPatches(patches:list,ptc:Patch):
    """Add a single patch to the list of patches list
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image:
    :param ptc: Single patch to be added to the list of patches list
    :return: Updated patches list
    :rtype: list
    """
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

def setConfidence(patches:list,camname:str,parkname:str,factor:float):
    """Set the confidence value for a patch in its own patch list (All patches of a single parking lot from all different parking lot image sources).
    The confidence value resulted is the multiplication of the old value times factor value.
    At the end, all confidence value are normalized in range 0 - 1.
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image:
    :param camname: Image file name (without extension) that produce the patch of that parking lot
    :parkname: Name parameter of :class:'Patch' corresponding to the park lot
    :return: Updated patches list
    :rtype: list
    """
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

def classify(model:keras.Model, shape:tuple, patches:list,verbose:int=0) -> dict:
    """From patches list, the method classify produces the classification output for each patch weighted for each confidence value.
    :param tuple:
    :param shape:
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image:
    :param verbose: Single patch to be added to the list of patches list
    :return: 
    :rtype: 
    """
    predResult = {}
    timeForClassific = 0
    for plist in patches:
        parkname = plist[0].name
        tempResult = []
        confs = []
        for patch in plist:
            prediction, timeReq = classifier.classify(model,shape,patch.imgArray)
            timeForClassific += timeReq
            pred = (prediction[0][0] - 0.5) * 2
            pred = pred * 100
            result = ""
            if pred > 0: result = "free"
            else: result = "occupied"
            classifResult = "{:.2f}% ".format(abs(pred)) + result
            tempResult.append(pred)
            confs.append(patch.confidence)
            if verbose: print(patch.name,"-",patch.imsource,":",classifResult,"[",patch.confidence,"]")
        predResult[parkname] = weightedMean(tempResult,confs)
    i = 0
    if verbose: 
        print()
        for parking in predResult:
            result = ""
            if predResult[parking] > 0: result = "free"
            else: result = "occupied"
            classifResult = "{:.2f}% ".format(abs(predResult[parking])) + result
            if verbose: print(patches[i][0].name,classifResult,"[",str(len(patches[i])),"cameras ]")
            i += 1
    print("Refresh rate:",secToTime(timeForClassific),end="\r",flush=True)
    return predResult

def createOverlay(imgArrayList:dict,patches:list,preds:dict,showImgList:list=[]):
    red_color = (0, 0, 255, 200)
    green_color = (0, 255, 0, 200)
    black_color = (20, 20, 20, 200)
    for imgArray in imgArrayList:
        rgb_data = imgArrayList[imgArray]
        rgba = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
        imgArrayList[imgArray] = rgba
    for imfile in imgArrayList:
        img = imgArrayList[imfile]
        for parkingPatches in patches:
            for patch in parkingPatches:
                if patch.imsource == imfile:
                    result = preds[patch.name]
                    if result>TOL:
                        color = green_color
                    elif result<-TOL: 
                        color = red_color
                    else: 
                        color = black_color
                    _x, _y = (patch.x1+5,patch.y2-5)
                    text = "{:.0f}% ".format(abs(result))
                    text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, LINE_TYPE)
                    text_w, text_h = text_size
                    cv2.rectangle(img, (_x-2,_y+2), ((_x + text_w), (_y - text_h)), darker(color), -1)
                    cv2.putText(img,text, 
                        (_x,_y), 
                        FONT, 
                        FONT_SCALE,
                        FONT_COLOR,
                        LINE_TYPE)
                    cv2.rectangle(img,(patch.x1,patch.y1),(patch.x2,patch.y2),color,3) 
        res = img
        cv2.imwrite("OUTPUTS/overlay_"+imfile,res)
        imgArrayList[imfile] = res
    for outfile in showImgList:
        try:
            outimg = imgArrayList[outfile]
        except KeyError:
            continue
        cv2.imshow("Overlay " + outfile,outimg)
        cv2.waitKey(1)

def listFiles(path:str,ext:str):
    filelist = []
    for file in os.listdir(path):
        if file.endswith(ext):
            filelist.append(file)
    return filelist

def readImgs(path:str,imgfiles:list):
    imgArrayList = {}
    for imgfile in imgfiles:
        imgArrayList[imgfile] = cv2.imread(path+"/"+imgfile)
    return imgArrayList

def parseRoots(path:str,xmlfiles:list):
    xmlRoots = {}
    for xmlfile in xmlfiles:
        xmlRoots[xmlfile] = ET.parse(path+"/"+xmlfile).getroot()
    return xmlRoots

def createPatches(xmlRoots:list,imgArrayList:dict):
    filelist = []
    patches = []
    for xmlfile in xmlRoots:
        filelist.append(getFName(xmlfile))
    for file in filelist:
        try:
            xmlElement = xmlRoots[file+".xml"]
            imgArray = imgArrayList[file+IMG_FORMAT]
        except KeyError:
            printErr("Cannot initialize patches because is not possible to match img files with corrispondent xml files\n"+
                + "Please be sure that xmls has the same names of img files. Image file format requested: " + IMG_FORMAT)
            sys.exit(1)
        patchname = ""
        x1, x2, y1, y2 = (0, 0, 0, 0)
        for elem in xmlRoots[file+".xml"]:
            for patch in elem:
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
            fullimg = imgArrayList[file+IMG_FORMAT]
            ptc = Patch(patchname,fullimg[y1:y2,x1:x2],file+IMG_FORMAT,x1,x2,y1,y2)
            patches = addToPatches(patches,ptc)
    return patches

def initialize():
    xmlpath = sys.argv[1]
    imgpath = sys.argv[2]
    imgsToShow = []
    for i in range(3,len(sys.argv)):
        imgsToShow.append(sys.argv[i])
    print(imgsToShow)
    xmlfiles = listFiles(xmlpath,".xml")
    imgfiles = listFiles(imgpath,IMG_FORMAT)
    imgArrayList = readImgs(imgpath,imgfiles)
    xmlRoots = parseRoots(xmlpath,xmlfiles)
    patches = createPatches(xmlRoots,imgArrayList)
    model, shape = classifier.loadModel(CLASSIFIER)
    return imgfiles, imgpath, model, shape, imgArrayList, imgsToShow,patches

def livePrediction(imgfiles:list,imgpath:str,model:keras.Model,shape:tuple,imgArrayList:dict,imgsToShow:list,patches:dict):
    global prediction_results
    while True:
        patches = updatePatches(imgfiles,imgpath,patches)
        prediction_results = classify(model, shape, patches)
        createOverlay(imgArrayList,patches,prediction_results,showImgList=imgsToShow)

def storePrediction(time_interval:int=5):
    time.sleep(10)
    global prediction_results
    while True:
        if prediction_results != {}:
            now = datetime.datetime.now()
            p0 = "JSON/LAST"
            p1 = "JSON/"+str(now.year)+"/"+str(now.month)+"/"+str(now.day)

            if not os.path.exists(p1): os.makedirs(p1)
            if not os.path.exists(p0): os.makedirs(p0)

            with open(p1+"/"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(now.microsecond)+".json", 'w') as outfile1:
                json.dump(prediction_results, outfile1)
            with open(p0+"/LAST.json", 'w') as outfile2:
                json.dump(prediction_results, outfile2)
        time.sleep(time_interval)
    

if __name__ == "__main__":
    #RULES: The name of XML file must match the name of image file
    imgfiles,imgpath,model,shape,imgArrayList,imgsToShow,patches = initialize()
    t = Thread(target=storePrediction, args=())
    t.start()
    livePrediction(imgfiles,imgpath,model,shape,imgArrayList,imgsToShow,patches)
    
    

    
