from CNNs import model_mAlexNet
from CNNs import model_AlexNet
from utilities import drawCurves

import numpy as np
import os
import cv2
import math

BSIZE = 64
EPOCHS = 5
DATADIR = "./DS_TEST/DATA"
LABELSDIR = "./DS_TEST/LABELS"
VALIDDIR = "./DS_TEST/VALID"
TESTDIR = "./DS_TEST/TEST"
LABFILE="labels.txt"

class Dataset:
    def  __init__(self,data,label):
       self.data = data
       self.label = label
    def  __init__(self,data):
       self.data = data

def getDataset():
    dset = Dataset([],[])
    lines = []
    with open(LABELSDIR+"/"+LABFILE,"r") as file:
        lines = file.readlines()
    for line in lines:
        x = line.split()
        img_array = cv2.imread(x[0])
        dset.data.append(img_array)
        dset.label.append(x[1])
    return dset

def getValidationDataset():
    dset = Dataset([],[])
    for img in os.listdir(VALIDDIR):
        img_array = cv2.imread(os.path.join(VALIDDIR,img))
        dset.data.append(img_array)
    return Dataset()

def train(model, trainData, validData):
    history = model.fit(trainData.data,trainData.label,validation_data=validData.data, batch_size=BSIZE, shuffle=True, verbose=1, epochs=EPOCHS)
    drawCurves(history)
    return history

def train(model, trainData, valid_splt):
    history = model.fit(trainData.data,trainData.label,validation_split=valid_splt, batch_size=BSIZE, shuffle=True, verbose=1, epochs=EPOCHS)
    drawCurves(history)
    return history

def testCNN():
    data, label = getDataset()
    model = model_mAlexNet() # (150x150x3)
    print("Model loaded succesfully")
    history = train(model,getDataset(),getValidationDataset())

def main():
    testCNN()

if __name__ == "__main__":
    main()