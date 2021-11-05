import cv2
import os
import sys
import xml.etree.ElementTree as ET
import keras
import json
import datetime
import time
import numpy
import PIL

from utilities import secToTime, darker, weightedMean, getFName
from tensorflow.python.eager.context import num_gpus
import classifier
from utilities import printErr
from threading import Thread
from skimage import io
from io import BytesIO


CLASSIFIER = "NETWORK/malexcnr.keras"
IMG_FORMAT = ".jpg"

TOL = 0.15
a = -100
b = 100

FONT_COLOR = (255,255,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
LINE_TYPE = 1

OCCUPIED_COLOR = (0, 0, 255, 200)
FREE_COLOR = (0, 255, 0, 200)
UNCERTAIN_COLOR = (20, 20, 20, 200)

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

def updatePatches(imgdirs:list,imgpath:str,patches:list) -> "tuple[list,dict]":
    """Update all image data for each patch in patches list
    :param imgdirs: List containing all image file names of all frames to be update
    :param imgpath: Path that contains all image files
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image.
    :return: Updated patches list and image dictionary
    """
    imgArrays = readImgs(imgdirs,imgpath)
    
    for patchList in patches:
        for patch in patchList:
            if imgArrays[patch.imsource] is None: continue
            patch.imgArray = imgArrays[patch.imsource][patch.y1:patch.y2,patch.x1:patch.x2]
    return patches, imgArrays

def addToPatches(patches:list,ptc:Patch) -> list:
    """Add a single patch to the list of patches list
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image.
    :param ptc: Single patch to be added to the list of patches list
    :return: Updated patches list
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

def setConfidence(patches:list,camname:str,parkname:str,factor:float) -> list:
    """Set the confidence value for a patch in its own patch list (All patches of a single parking lot from all different parking lot image sources).
    The confidence value resulted is the multiplication of the old value times factor value.
    At the end, all confidence value are normalized in range 0 - 1.
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image.
    :param camname: Image file name (without extension) that produce the patch of that parking lot
    :parkname: Name parameter of :class:'Patch' corresponding to the park lot
    :return: Updated patches list
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

def swap(a,b):
    a = t
    a = b
    b = t
    return a,b

def classify(model:keras.Model, shape:tuple, patches:list,verbose:int=0) -> dict:
    global a,b
    """From patches list, the method classify produces the classification output for each patch weighted for each confidence value.
    :param model: The trained Keras classificator model able to do the parking classification
    :param shape: Image input shape as input of classificator (Example: (150,150,3) for 150x150 pixel RGB images)
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image.
    :param verbose: 0 or 1 if printing classification result is needed
    :return: Dictionary cointains for each park ID (key) the classification value (value) between a=-100 and b=100 (global default values)
    """
    if a > b: a,b = swap(a,b)
    if a==b: b += 1
    predResult = {}
    timeForClassific = 0
    for plist in patches:
        parkname = plist[0].name
        tempResult = []
        confs = []
        for patch in plist:
            prediction, timeReq = classifier.classify(model,shape,patch.imgArray)
            timeForClassific += timeReq
            factor = b - a
            transl = a + (factor/2)
            pred = (prediction[0][0] - 0.5 + transl) * factor
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
            if predResult[parking] > (b-a)/2: result = "free"
            else: result = "occupied"
            classifResult = "{:.2f}% ".format(abs(predResult[parking])) + result
            if verbose: print(patches[i][0].name,classifResult,"[",str(len(patches[i])),"cameras ]")
            i += 1
    print("Refresh rate:",secToTime(timeForClassific),end="\r",flush=True)
    return predResult

def createOverlay(imgDictionary:dict,patches:list,preds:dict,showImgList:list=[],postfix:str="%"):
    """From the full image dictionary, patch list and prediction creates the output image as sum of original image and a box overlay.
    Each patch match a green box (if parking is free), a red box (if parking is occupied) or a black box if the classification is between a treshold value (golbal TOL variable).
    Every box carry the classification value inside.
    :param imgDictionary: original full park lot image dictionary
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image
    :param preds: Dictionary cointains for each park ID (key) the classification value (value) between a=-100 and b=100 (global default values)
    :param showImgList: List of names of the images to be shown during prediction with overlay applied. If image name is wrong program simply will not show it.
    :param postfix: A string applied at the end of classification result printed onto classification box
    """
    for imgArray in imgDictionary:
        rgb_data = imgDictionary[imgArray]
        rgba = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
        imgDictionary[imgArray] = rgba
    for imfile in imgDictionary:
        img = imgDictionary[imfile]
        for parkingPatches in patches:
            for patch in parkingPatches:
                if patch.imsource == imfile:
                    result = preds[patch.name]
                    if result> ((b-a)/2) + (b - a)*TOL:
                        color = FREE_COLOR
                    elif result< ((b-a)/2) - (b - a)*TOL: 
                        color = OCCUPIED_COLOR
                    else: 
                        color = UNCERTAIN_COLOR
                    _x, _y = (patch.x1+5,patch.y2-5)
                    text = "{:.0f}".format(abs(result))+postfix
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
        imgDictionary[imfile] = res
    for outfile in showImgList:
        try:
            outimg = imgDictionary[outfile]
        except KeyError:
            continue
        cv2.imshow("Overlay " + outfile,outimg)
        cv2.waitKey(1)


def listFiles(path:str,ext:str) -> list:
    """List files contained in the specified directory that match the specified extension
    :param path: Directory path
    :param ext: File extension
    :return: List of files with that extension
    """
    filelist = []
    for file in os.listdir(path):
        if file.endswith(ext):
            filelist.append(file)
    return filelist

def readImgs(imgfiles:list,imgpath:str) -> dict:
    imgDictionary = {}
    """Get image pixel matrix from a list of files. If image file get a premature end the function keep trying reading image until image is readable
    :param imgfiles: List of image file names to be loaded
    :param imgpath: Path of image files 
    :return: Dictionary of image files (name: key) containing the pixel matrix of each image (value)
    """
    for imgfile in imgfiles:
        img = None
        while True:
            try:
                with open(imgpath+"/"+imgfile, 'rb') as img_bin:
                    buff = BytesIO()
                    buff.write(img_bin.read())
                    buff.seek(0)
                    temp_img = numpy.array(PIL.Image.open(buff), dtype=numpy.uint8)
                    img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                if img is not None:
                    break
            except OSError:
                continue
        imgDictionary[imgfile] = img

    return imgDictionary

def parseRoots(xmlfiles:list,path:str)-> dict:
    """Get patch children defined in a list of XML files
    :param xmlfiles: List of XML file names to be parsed
    :param path: Path of XML files
    :return: Dictionary of XML files (name: key) containing the parsed root of each patch (value). 
    Same parking spots of different XML files should share the same parking ID.
    """
    xmlRoots = {}
    for xmlfile in xmlfiles:
        xmlRoots[xmlfile] = ET.parse(path+"/"+xmlfile).getroot()
    return xmlRoots

def createPatches(xmlRoots:dict,imgDictionary:dict):
    """Initialize patches defined in the XML files and create a list of patch list: every member of the list represents all parking patch taken from different images.
    :param xmlRoots: Dictionary of XML files (name: key) containing the parsed root of each patch (value)
    :param imgDictionary: Dictionary of image files (name: key) containing the pixel matrix of each image (value)
    :return: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image
    """
    filelist = []
    patches = []
    for xmlfile in xmlRoots:
        filelist.append(getFName(xmlfile))
    for file in filelist:
        try:
            _ = xmlRoots[file+".xml"]
            _ = imgDictionary[file+IMG_FORMAT]
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
            fullimg = imgDictionary[file+IMG_FORMAT]
            ptc = Patch(patchname,fullimg[y1:y2,x1:x2],file+IMG_FORMAT,x1,x2,y1,y2)
            patches = addToPatches(patches,ptc)
    return patches

def initialize()->"tuple(list,str,keras.Model,tuple,dict,list,list)":
    """Parse main params: arg 1 is the path containing XML patch file, arg 2 is the path containing correspondent images, arg 3 is the list of images to show with classification overlay.
    :return: List of image file names, image path containing these images, Keras trained classificator model, image shape requested, image dictionary, list of image to show, list of patches list
    """
    xmlpath = sys.argv[1]
    imgpath = sys.argv[2]
    if (xmlpath is None) or (imgpath is None) :
        print("Usage: pkspot.py [XMLPATH] [IMGPATH] [LIST OF IMAGE NAMES TO SHOW OVERLAY] ...")
        sys.exit(1)
    imgsToShow = []
    for i in range(3,len(sys.argv)):
        imgsToShow.append(sys.argv[i])
    xmlfiles = listFiles(xmlpath,".xml")
    imgfiles = listFiles(imgpath,IMG_FORMAT)
    imgDictionary = {}
    x = 1
    while(x):
        try:
            imgDictionary = readImgs(imgfiles,imgpath)
            x = 0
        except ValueError:
            continue
    xmlRoots = parseRoots(xmlfiles,xmlpath)
    patches = createPatches(xmlRoots,imgDictionary)
    model, shape = classifier.loadModel(CLASSIFIER)
    return imgfiles, imgpath, model, shape, imgDictionary, imgsToShow,patches

def livePrediction(imgfiles:list,imgpath:str,model:keras.Model,shape:tuple,imgDictionary:dict,imgsToShow:list,patches:list):
    """Start the prediction cycle
    :param imgfiles: List of image file names to be loaded
    :param imgpath: Path of image files
    :param model: The trained Keras classificator model able to do the parking classification
    :param shape: Image input shape as input of classificator (Example: (150,150,3) for 150x150 pixel RGB images)
    :param imgDictionary: Dictionary of image files (name: key) containing the pixel matrix of each image (value)
    :param imgsToShow: List of names of the images to be shown during prediction with overlay applied. If image name is wrong program simply will not show it.
    :param patches: List of patch list. Every patch list contains all patches of a single parking spot from all different parking lot source image
    """
    global prediction_results
    while True:
        try:
            patches, imgDictionary = updatePatches(imgfiles,imgpath,patches)
            prediction_results = classify(model, shape, patches)
            createOverlay(imgDictionary,patches,prediction_results,showImgList=imgsToShow)
        except ValueError:
            continue

def storePrediction(time_interval:int=900):
    """Store prediction dictionary as JSON. Every json is stored every fixed amount of time and categorized by date and time
    :param time_interval: Fixed dumping time in second (default 900s : 15 mins)
    """
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
    imgfiles,imgpath,model,shape,imgDictionary,imgsToShow,patches = initialize()
    t = Thread(target=storePrediction, args=())
    t.start()
    livePrediction(imgfiles,imgpath,model,shape,imgDictionary,imgsToShow,patches)
    
    

    
