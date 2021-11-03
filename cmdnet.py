#!/usr/bin/env python3

VERSION = "1.0.4.6"

from utilities import isBoolean, listFileInDir, summarizeAccuracy, summarizeLoss, qry, toBool, imgpad, secToTime, histEq, printErr, bcolors
from keras.backend import int_shape
from CNNs import model_VGG16, model_leNet, model_mAlexNet
from CNNs import model_AlexNet
from utilities import drawCurves, printProgressBar, reshape_image, printBorder, isFloat, isInt
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from sklearn import metrics
from keras import backend as K

import tensorflow as tf
import keras
import numpy as np
import os
import cv2
import time
import sys
import socket
import cmd
import random
import math



class Dataset:
    def  __init__(self,data,label):
       self.data = data
       self.label = label

class Reshaper:
    def __init__(self,mod,val):
        self.mod = mod
        self.val = val

class TimeHistory(keras.callbacks.Callback):
    def reset(self):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class DynamicLR(keras.callbacks.Callback):
    epochCount = 0
    epo = 2
    val = 0.75
    def __init__(self, epo, val):
        self.epo = epo
        self.val = val
    def on_epoch_end(self,batch,logs={}):
        self.epochCount +=1
        if self.epochCount == self.epo:
            self.epochCount = 0
            K.set_value(model.optimizer.learning_rate, model.optimizer.learning_rate*self.val)
            print("Learning rate changed to: ", K.get_value(model.optimizer.learning_rate))

def setLr(lr):
    try:
        K.set_value(model.optimizer.learning_rate, lr)
        updateLr()
    except:
        printErr("Unable to change learning rate value!")
        return

def updateLr():
    global lrate
    lrate = K.get_value(model.optimizer.learning_rate)

interpmod = {
    cv2.INTER_LINEAR : "linear",
    cv2.INTER_CUBIC : "cubic",
    cv2.INTER_NEAREST : "nearest",
    cv2.INTER_LANCZOS4 : "lanczos - 4"
}

ENV_NAME = socket.gethostname()

ROW = 0
COL = 0
CH = 0

ABSPATH= os.path.dirname(__file__)

BASEPATH = ""
LABELS_DATA = ""
LABELS_VALID = ""
LABELS_TESTSET = ""

#Default values:
dsname = "None"
history = -1
bsize = 64
rseed = 4
epochs = 11
shuffle = 1
model = None
model_trained = 0
mname = "None"
training_time = 0
lrate = 0.0008
edsr = ""
equalize = 1

first = 1
original_weights = []

dlr = 1
everyepoch = 2
lrf = 0.75

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,restore_best_weights=True)
time_callback = TimeHistory()
dynamicTrain = DynamicLR(everyepoch,lrf)
reshaper = Reshaper("skip",cv2.INTER_CUBIC)


def setDynamicLr():
    global dynamicTrain
    dynamicTrain = DynamicLR(everyepoch,lrf)

def setSeed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setDataset(pathToFile):
    global BASEPATH
    global LABELS_DATA
    global LABELS_VALID
    global LABELS_TESTSET
    lines = []
    if not os.path.isfile(os.path.join(ABSPATH,pathToFile)):
        printErr("File not found at: "+ os.path.join(ABSPATH,pathToFile))
        return 0
    with open(os.path.join(ABSPATH,pathToFile),"r") as file:
        lines = file.readlines()
    if len(lines) != 4:
        printErr("Error on reading dataset specified on file: "+ pathToFile)
        return 0
    BASEPATH = lines[0].rstrip("\n")
    LABELS_DATA = lines[1].rstrip("\n")
    LABELS_VALID = lines[2].rstrip("\n")
    LABELS_TESTSET = lines[3].rstrip("\n")
    return 1

def getModel(path, fname):
    global model, ROW, COL, CH, mname, model_trained
    model = ""
    model = keras.models.load_model(path)
    print("Model ", path, " loaded succesfully")
    ROW = model.layers[0].get_input_at(0).get_shape()[1]
    COL = model.layers[0].get_input_at(0).get_shape()[2]
    CH = model.layers[0].get_input_at(0).get_shape()[3]
    mname = fname
    model_trained = 1

def getDataset(pathToFile,basepath,verbose = 1,inspect=0,rewrite=0):
    global reshaper, edsr
    loaded_index = 0
    loaded_total = 0
    file_skipped = 0
    file_converted = 0
    lines = []
    if not os.path.isfile(os.path.join(ABSPATH,pathToFile)):
        printErr("File not found at: "+ os.path.join(ABSPATH,pathToFile))
        return 0
    with open(os.path.join(ABSPATH,pathToFile),"r") as file:
        lines = file.readlines()
    loaded_total = len(lines)
    dset = Dataset([],[])
    if verbose: print("Loading data set defined in: ", pathToFile)
    if verbose: print("Total file count: ", loaded_total)
    loaded_index = 0
    total_index = 0
    postfix = []
    ETA = "-"
    tic = 1
    tt1 = time.time()

    acc = 0
    timeindex = 0
    prev_acc = np.Infinity
    
    for line in lines:

        t1 = time.time()
        timeindex += 1
        skipped = 0
        total_index += 1
        x = line.split()
        pth = os.path.join(basepath,x[0])
        imgname = x[0].split("/")[-1]
        img_array = cv2.imread(pth)
        if img_array is None:
            postfix.append("Cannot open image: " + os.path.abspath(pth))
            file_skipped += 1
            skipped = 1
        else:
            if equalize: img_array = histEq(img_array)
            if inspect: 
                cv2.imshow(pth,img_array)
                xx = pth.split("/")
                cv2.waitKey()
                cv2.imwrite(xx[-1],img_array)
            if img_array.shape != (ROW,COL,CH):
                if reshaper.mod == "padding":
                    img_array = imgpad(img_array,ROW,COL)
                    postfix.append("Reshaping image: " + imgname)
                    file_converted += 1
                elif reshaper.mod == "scalepadding":
                    _row,_col,_ch = img_array.shape
                    if _row >= _col:
                        col = math.ceil(ROW/_row*_col)
                        img_array = cv2.resize(img_array,(ROW,col), reshaper.val)
                    else:
                        row = math.ceil(COL/_col*_row)
                        img_array = cv2.resize(img_array,(row,COL), reshaper.val)
                    img_array = imgpad(img_array,ROW,COL)
                    postfix.append("Reshaping image: " + imgname)
                    file_converted += 1
                elif reshaper.mod.startswith("EDSR"):
                    img_array = edsr.upsample(img_array)
                    img_array = cv2.resize(img_array,(ROW,COL), reshaper.val)
                    postfix.append("Reshaping image: " + imgname)
                    if rewrite: 
                        cv2.imwrite("./"+basepath+"/"+reshaper.mod+"/"+x[0],img_array)
                        print("Saved to: " + "./"+basepath+"/"+reshaper.mod+"/"+x[0])
                    file_converted += 1
                elif reshaper.mod == "interpolate":
                    img_array = cv2.resize(img_array,(ROW,COL), reshaper.val)
                    postfix.append("Reshaping image: " + imgname)
                    file_converted += 1
                else:
                    postfix.append("Skipping image: " + imgname)
                    file_skipped += 1
                    skipped = 1
        if not skipped:
            dset.data.append(img_array)
            dset.label.append(int(x[1]))
        loaded_index += 1
        
        if inspect: 
                cv2.imshow(pth,img_array)
                xx = pth.split("/")
                cv2.waitKey()
                cv2.imwrite(xx[-1],img_array)

        t2 = time.time()
        acc += (t2-t1)
        if acc >= 1:
            eta = math.ceil(acc*((loaded_total-loaded_index)/timeindex))
            if eta < prev_acc:
                prev_acc = eta
                ETA = secToTime(eta)
            else:
                neta = (eta - prev_acc)/2
                prev_acc += neta
            
            timeindex = 0
            acc = 0
        suf = "loaded: " + str(loaded_index-file_skipped) + "/" + str(loaded_total) + " - skipped: " + str(file_skipped) + " - converted: " + str(file_converted) + " - ETA: " + str(ETA)
        if verbose: printProgressBar(total_index,loaded_total,suffix=suf,length=20)
    if verbose:
        print()
        print()
        printBorder("Total file count: " + str(loaded_total) +
        "\nFile loaded: " + str(loaded_total-file_skipped) + 
        "\nFile skipped: " + str(file_skipped) +
        "\nFile converted: " + str(file_converted))
        print()
        print("Converting data to numpy array: ")
    dset.data = np.array(dset.data)
    dset.label = np.array(dset.label)
    dset.label = to_categorical(dset.label,num_classes=2)
    tt2 = time.time()
    if verbose: 
        print("Conversion completed")
        print("Data shape: ", dset.data.shape)
        print("Labels shape: ", dset.label.shape)
        print("Total time for dataset loading: " + secToTime((tt2-tt1)))
    if (loaded_total-file_skipped) == 0: return -1
    return dset

def saveData(res,prefix="TEST_RESULT"):
    folder = "./NETWORK/RESULT/"+mname
    if qry("Save " + prefix + " into " + folder +" ? "):
        if not os.path.exists(folder):
            os.makedirs(folder)
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d_%b_%Y(%H_%M_%S.%f)")
        filename = prefix+"_"+mname+"_"+timestampStr+".log"
        with open(folder+"/"+filename, "x") as text_file:
            text_file.write(getSummary()+"\n"+res)
            print("Results saved as: ", filename)

def train(model,bsize,shuffle,epochs,trainData, validData):
    global history, training_time, time_callback
    print()
    print("Training is starting with input shape: ", trainData.data.shape)
    print()
    time_callback.reset()
    callbacks = [time_callback]
    if dlr: callbacks.append(dynamicTrain)
    if early_stopping: callbacks.append(early_stopping)
    history = model.fit(x=trainData.data,y=trainData.label,validation_data=(validData.data,validData.label), batch_size=bsize, shuffle=shuffle, verbose=1, epochs=epochs,callbacks=callbacks)
    training_time = sum(time_callback.times)
    return history

def trainWithValSplit(model,bsize,shuffle,epochs,trainData, valid_splt):
    global history, training_time, time_callback
    print()
    print("Training is starting with input shape: ", trainData.data.shape)
    print()
    time_callback.reset()
    history = model.fit(trainData.data,trainData.label,validation_split=valid_splt, batch_size=bsize, shuffle=shuffle, verbose=1, epochs=epochs,callbacks=[time_callback])
    training_time = sum(time_callback.times)
    return history

def liveTest(model,testSet,savequery=1):
    for _img in testSet.data:
        img = np.array([_img])
        prd = model.predict(img,verbose=1)
        if prd[0][0]<0.5: print("Occupato al {:.2f} %".format((prd[0][1])*100))
        else: print("Libero al {:.2f} %".format((prd[0][0])*100))
        cv2.imshow("Immagine",_img)
        cv2.waitKey()

def test(model,testSet,savequery=1):
    print("Testing: ")
    t_start = time.time()
    pred = model.predict(testSet.data,verbose=1)
    t_end = time.time()
    print("Test done")
    tn, fp, fn, tp = metrics.confusion_matrix(pred.argmax(axis=1), testSet.label.argmax(axis=1)).ravel()
    p = tp + fn
    n = fp + tn
    acc = (tp + tn) / (p+n)
    f1 = (2*tp)/((2*tp)+fp+fn) #f1 score
    tpr, tnr, fnr, fpr = tp/p, tn/n, fn/p, fp/n
    res = "P: " + str(p) + "\n" +\
        "N: " + str(n) + "\n" +\
        "TN: " + str(tn) + "\n" +\
        "TP: " + str(tp) + "\n" +\
        "FN: " + str(fn) + "\n" +\
        "FP: " + str(fp) + "\n" +\
        "\n" +\
        "Accuracy: " + str(acc*100) + "%\n" +\
        "F1 score: " + str(f1) + "\n" +\
        "\n" +\
        "TNR: " + str(tnr) + "\n" +\
        "TPR: " + str(tpr) + "\n" +\
        "FNR: " + str(fnr) + "\n" +\
        "FPR: " + str(fpr) + "\n" 
    tim = secToTime(t_end-t_start)
    results = "\n"+res+"\nTest execution time: " + str(tim)
    printBorder(getSummary()+results)
    if not savequery:
        return results, (acc, f1, tpr, tnr, fnr, fpr)
    saveData(results)
    return results

def getSummary():
    shuf = "Yes" if shuffle else "No"
    tr = "trained" if model_trained else "not_trained"
    dlrstr = str(lrate) + " * " + str(lrf) + " every " + str(everyepoch) + " epoch(s)" if dlr else str(lrate)
    resh = ""
    if reshaper.mod == "skip": resh ="No"
    elif reshaper.mod == "padding": resh ="Padding"
    elif reshaper.mod == "interpolate": resh="Interpolation: " + interpmod[reshaper.val]
    elif reshaper.mod == "scalepadding": resh="Scaled - padding: " + interpmod[reshaper.val]
    elif reshaper.mod.startswith("EDSR"): resh="Super resolution " + reshaper.mod
    smry = "Dataset selected: "+ str(dsname) +"\n"+\
        "Network: " + str(mname) + "\n" +\
        "Network trained: " + str(tr) + "\n" +\
        "Batch size: " + str(bsize) + "\n" +\
        "Epoch(s): " + str(epochs) + "\n" +\
        "Random seed: " + str(rseed) + "\n" +\
        "Shuffle: " + str(shuf) + "\n" +\
        "Reshape image: " + str(resh) + "\n" +\
        "Learning rate: " + dlrstr + "\n" +\
        "Image shape requested: (" + str(ROW) + "x" + str(COL) + "x" + str(CH) + ")\n"
    return smry

def selector(name,path,ext,selection):
    list = listFileInDir(path,ext,prnt=0)
    if not selection:
        print("Avaliable " + name + ": ")
        listFileInDir(path,ext,prnt=1)
        return 0, ""
    if isInt(selection):
        if (int(selection)-1)>=len(list) or int(selection)<=0:
            print("Avaliable " + name + ": ")
            listFileInDir(path,ext,prnt=1)
            return 0, ""
        file_selected = list[int(selection)-1]
        selection = file_selected
    if os.path.isfile(path+"/"+selection):
        return 1, path+"/"+selection
    else: 
        printErr("Cannot open file: "+str(path)+"/"+str(selection))
        return 0, ""


import cmd
commands = []

class CmdParse(cmd.Cmd):
    prompt = bcolors.BOLD + bcolors.OKCYAN + " (cmdnet) " + bcolors.OKGREEN + ENV_NAME + " >> " + bcolors.ENDC + "\033[0m"
    
    def do_listall(self, line):
        print(commands)
    def do_summary(self, line):
        printBorder(getSummary())
    def do_dataset(self,dset_selected,verbose=1):
        global dsname
        x,ds_name = selector("datasets","./DATASET","ds",dset_selected)
        if not x: return
        check = setDataset(ds_name)
        if not check: 
            printErr("Unable to load dataset: " + str(dset_selected))
            return
        dsname=ds_name
        if verbose: print("Dataset selected: ", ds_name)
    def do_load(self, fname):
        x,m_name = selector("networks","./NETWORK","keras",fname)
        if not x: return
        name = m_name.split("/")[-1]
        getModel(m_name,name)
        updateLr()
    def do_new(self, fname, verbose=0):
        x,m_name = selector("models","./MODELS","h5",fname)
        if not x: return
        name = m_name.split("/")[-1]
        getModel(m_name,name)
        setLr(lrate)
    def do_ds(self,dset_selected):
        self.do_dataset(dset_selected)
    def do_lr(self,lrt):
        global lrate
        if lrt:
            if isFloat(lrt):
                lrate = float(lrt)
                setLr(lrate)
                print("Learning rate changed to: ", lrate)
            else:
                printErr("Error on parsing value as integer: " + str(lrt))
        else: print("Learning rate: ", lrate)
    def do_batch(self,batch):
        global bsize
        if batch:
            if isInt(batch):
                bsize = int(batch)
                print("Batch size changed to: ", bsize)
            else:
                printErr("Error on parsing value as integer: " + str(batch))
        else: print("Batch size: ", bsize)
    def do_seed(self,sd):
        global rseed
        if sd:
            if isInt(sd):
                rseed = int(sd)
                setSeed(rseed)
                print("Seed changed to: ", rseed)
            else:
                print("Error on parsing value as integer: ", sd)
        else: print("Seed: ", rseed)
    def do_epoch(self,ep):
        global epochs
        if ep:
            if isInt(ep):
                epochs = int(ep)
                print("Epoch(s) changed to: ", epochs)
            else:
                printErr("Error on parsing value as integer: " + str(ep))
        else: print("Epoch(s): ", epochs)
    def do_earlystop(self,line):
        global early_stopping
        if line:
            line = toBool(line)
            if line != -1:
                es = int(line)
                if es == 0: 
                    early_stopping = None
                    es = "false"
                else:
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,restore_best_weights=True)
                    es = "true"
                print("Early stopping changed to: ", es)
            else:
                printErr("Error on parsing value as boolean: " + str(line))
        else:
            if early_stopping: print("Early stopping: true")
            else: print("Early stopping: false")
    def do_shuffle(self,shuf):
        global shuffle
        if shuf:
            shuf = toBool(shuf)
            if shuf != -1:
                sh = int(shuf)
                if sh == 0: 
                    shuffle = 0
                    sh = "false"
                else:
                    shuffle = 1
                    sh = "true"
                print("Shuffle changed to: ", sh)
            else:
                printErr("Error on parsing value as boolean: " + str(shuf))
        else:
            if shuffle: print("Shuffle: true")
            else: print("Shuffle: false")
    def do_reshape(self,args):
        global reshaper, edsr
        params = args.split()
        if len(params) == 0:
            resh = ""
            if reshaper.mod == "skip": resh ="No"
            elif reshaper.mod == "padding": resh ="Padding"
            elif reshaper.mod == "interpolate": resh="Interpolation: " + interpmod[reshaper.val]
            elif reshaper.mod == "scalepadding": resh="Scaled - padding: " + interpmod[reshaper.val]
            elif reshaper.mod.startswith("EDSR"): resh="Super resolution " + reshaper.mod
            print("Reshape policy: " + resh)
        else:
            if params[0].lower() == "s" or params[0].lower() == "skip":
                reshaper.mod = "skip"
                print("Reshape policy changed to: skip")
                return
            if params[0].lower() == "interpolate" or params[0].lower() == "scalepadding":
                if len(params) == 1:
                    print("Wrong value for reshape policy. You must specify one of the following interpolation algorithms:")
                    print("linear")
                    print("cubic")
                    print("nearest")
                    print("lanczos")
                    return
                if params[1].lower() == "linear":
                    reshaper.mod = params[0].lower()
                    reshaper.val = cv2.INTER_LINEAR
                    print("Reshape policy changed to: "+params[0].lower()+" linear")
                elif params[1].lower() == "cubic":
                    reshaper.mod = params[0].lower()
                    reshaper.val = cv2.INTER_CUBIC
                    print("Reshape policy changed to: "+params[0].lower()+" cubic")
                elif params[1].lower() == "nearest":
                    reshaper.mod = params[0].lower()
                    reshaper.val = cv2.INTER_NEAREST
                    print("Reshape policy changed to: "+params[0].lower()+" nearest")   
                elif params[1].lower() == "lanczos":
                    reshaper.mod = params[0].lower()
                    reshaper.val = cv2.INTER_LANCZOS4
                    print("Reshape policy changed to: "+params[0].lower()+" lanczos")
                else: 
                    print("Wrong value for reshape policy. You must specify one of the following interpolation algorithms:")
                    print("linear")
                    print("cubic")
                    print("nearest")
                    print("lanczos")
            elif params[0].lower() == "p" or params[0].lower() == "padding":
                reshaper.mod = "padding"
                print("Reshape policy changed to: padding")
                return
            elif params[0].lower() == "sr" or params[0].lower() == "super resolution":
                reshaper.val = cv2.INTER_CUBIC
                edsr = cv2.dnn_superres.DnnSuperResImpl_create()
                _rfac = 4
                rfac = 4
                if len(params) < 2:
                    edsr.readModel("SR/EDSR_x"+_rfac+".pb")
                    edsr.setModel("edsr",_rfac)
                    reshaper.mod = "EDSRx"+_rfac
                else:
                    if isInt(params[1]):
                        rfac = int(params[1])
                        if rfac == 2 or rfac==3 or rfac == 4:
                            edsr.readModel("SR/EDSR_x"+str(rfac)+".pb")
                            edsr.setModel("edsr",rfac)
                            reshaper.mod = "EDSRx"+str(rfac)
                        else: 
                            printErr("You must select 2, 3 or 4 as scale factor. Default value "+str(_rfac)+" will be used")
                            edsr.readModel("SR/EDSR_x"+str(_rfac)+".pb")
                            edsr.setModel("edsr",_rfac)
                            reshaper.mod = "EDSRx"+str(_rfac)
                    else:
                        printErr("Unable to parse integer value as scale factor: " + params[1] + "\nAccepted values: 2, 3, 4")
                print("Reshape policy changed to: EDSR Super resolution x"+str(rfac))
                return
            else:
                print("Wrong value for reshape policy. Use 's' for skip, 'interpolate' for interpolate, 'p' for padding, 'sr' for EDSR super resolution or 'scalepadding' for scaled padding")
    def do_train(self,vspl):
        global model,history,model_trained,skip,bsize,shuffle,epochs
        if model is None:
           print("A network model is not selected. You can create a new one by typing command \'new\'")
           return
        if BASEPATH == "":
            print("Dataset not selected. Please chose one by typing command 'dataset'")
            return
        self.do_summary("")
        if not qry("Start training? "):
            return
        if vspl:
            if isFloat(vspl):
                ds = getDataset(LABELS_DATA,BASEPATH)
                if ds != 0 and ds != -1:
                    history = trainWithValSplit(model,bsize,shuffle,epochs,ds,float(vspl))
                    model_trained = 1
                    val, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
                    model_trained = 1
                    print("Train completed. Best validation loss value found on epoch: ", str(idx+1))
                else: 
                    printErr("Error while loading dataset")
                    return
            else:
                printErr("Error on parsing value as a float: " + str(vspl))
                return
        else:
            ds = getDataset(LABELS_DATA,BASEPATH)
            if ds == -1:
                printErr("Dataset empty")
                return
            vs = getDataset(LABELS_VALID,BASEPATH)
            if vs == -1:
                printErr("Dataset empty")
                return
            history = train(model,bsize,shuffle,epochs,ds,vs)
            val, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
            model_trained = 1
            print("Train completed. Best validation loss value found on epoch: ", str(idx+1))
    def do_test(self,testfile):
        global model_trained,dsname,LABELS_TESTSET
        if model is None:
            print("Dataset not selected. Please select it first by typing command \'dataset\'")
            return
        if not model_trained:
            print("Model is not trained. Please train it first by typing command \'train\'")
            return
        if testfile:
            testSet = getDataset(testfile,BASEPATH)
            if testSet == 0 or testSet == -1: 
                printErr("Error while loading test set: " + str(testfile))
                return
            test(model,testSet)
        else:
            if qry(LABELS_TESTSET + " will be used. It\'s ok?"):
                testSet = getDataset(LABELS_TESTSET,BASEPATH)
                if testSet == 0 or testSet == -1: 
                    printErr("Error while loading test set: " + str(testfile))
                    return
                test(model,testSet)
            else: return
    def do_livetest(self,testfile):
        global model_trained,dsname,LABELS_TESTSET
        if model is None:
            print("Dataset not selected. Please select it first by typing command \'dataset\'")
            return
        if not model_trained:
            print("Model is not trained. Please train it first by typing command \'train\'")
            return
        if testfile:
            testSet = getDataset(testfile,BASEPATH)
            if testSet == 0 or testSet == -1: 
                printErr("Error while loading test set: " + str(testfile))
                return
            liveTest(model,testSet)
        else:
            if qry(LABELS_TESTSET + " will be used. It\'s ok?"):
                testSet = getDataset(LABELS_TESTSET,BASEPATH)
                if testSet == 0 or testSet == -1: 
                    printErr("Error while loading test set: " + str(testfile))
                    return
                liveTest(model,testSet)
            else: return
    def do_dlr(self,lr):
        global dlr, lrf, everyepoch
        lrl = lr.split()
        if lr:
            if len(lrl)==1: 
                lr = toBool(lr)
                if lr != -1:
                    x = int(lr)
                    if x == 0: 
                        dlr = 0
                        x = "false"
                    else:
                        dlr = 1
                        x = "true"
                    print("Dynamic learning rate changed to: ", x)
            else:
                if len(lrl)!=2:
                    printErr("Dynamic learning rate format: dlr 'factor' 'epochs'")
                    return
                if not isFloat(lrl[0]) or not isInt(lrl[1]):
                    printErr("Dynamic learning rate format: dlr 'factor(float)' 'epochs(int)'")
                    return
                lrf = float(lrl[0])
                everyepoch = int(lrl[1])
                setDynamicLr()
        dlrstr = "lr * " + str(lrf) + " every " + str(everyepoch) + " epoch(s)" if dlr else "No"
        print("Dynamic learning rate: " + dlrstr)
    def do_save(self,fname):
        global mname, model, history, training_time
        if model is None:
            print("Model not instanced. Do it by typing command \'new\'")
            return
        if model_trained == 0:
            print("Model not trained. Do it by typing command \'train\'")
            return
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d_%b_%Y(%H_%M_%S.%f)")
        if fname: newname = fname+".keras"
        else: newname = timestampStr+"_"+mname+".keras"
        pth="NETWORK/"+newname
        model.save(pth)
        mname = newname
        print("Model saved: ",pth)
        tr_accuracy = str(history.history["accuracy"]) + "\n"
        tr_vacc = str(history.history["val_accuracy"]) + "\n"
        tr_loss = str(history.history["loss"]) + "\n"
        tr_vloss = str(history.history["val_loss"]) + "\n"
        tr_time = secToTime(training_time)
        val, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
        saveData("HISTORY TRAINING DATA:\nEpochs: "+str(len(history.history["accuracy"]))+"\nBest Valid. loss value: "+ str(val) +" at epoch: "+str(idx+1)+"\n\nAccuracy:\n" +tr_accuracy+"\nLoss\n"+tr_loss+ "\nValid. accuracy\n"+
            tr_vacc+"\nValid. loss\n"+tr_vloss+"\n\nTRAINING TIME: "+str(tr_time),"TRAINING_RESULT")
    def do_plot(self, grf):
        if history == -1:
            print("History not available for this model. Please train the model first")
            return
        if grf:
            if grf == "a" or grf =="accuracy": 
                summarizeAccuracy(history,mname)
            elif grf == "l" or grf =="loss": 
                summarizeLoss(history,mname)
            else: print("Specify if you want to plot \'a\': accuracy, or \'l\': loss")
        else: print("Specify if you want to plot \'a\': accuracy, or \'l\': loss")
    def do_multirun(self,line):
        global model, bsize, shuffle, epochs, history,rseed
        x,pth = selector("run files","./RUN","run",line)
        if not x: return
        with open(pth,"r") as file:
            lines = file.readlines()
        if len(lines) != 5:
            printErr("File: " + str(pth) + " is not well formatted")
            return
            
        inseed = int(lines[0].rstrip("\n"))
        finseed = int(lines[1].rstrip("\n"))
        rseed = inseed
        self.do_seed(inseed)
        netmodel = lines[2].rstrip("\n")
        dstrain = lines[3].rstrip("\n")
        dstest = lines[4].rstrip("\n")
        runlength = finseed - inseed
        self.do_new(netmodel,verbose=0)
        self.do_dataset(dstrain,verbose=0)
        smry = "RUN(s) SIZE: " + str(runlength) + "\n"
        smry += getSummary()
        printBorder(smry)
        if not qry("Proceed with run? "):
            return    
        dset = getDataset(LABELS_DATA,BASEPATH)
        vset = getDataset(LABELS_VALID,BASEPATH)
        self.do_dataset(dstest, verbose=0)
        tset = getDataset(LABELS_TESTSET,BASEPATH)
        results = ""
        results += "DATASET: " + LABELS_DATA + "\n"
        results += "VALIDATION SET: " + LABELS_VALID  + "\n"
        results += "TEST SET: " + LABELS_TESTSET  + "\n\n"
        runindex = 1
        testresvalues = []
        avtimetr = 0
        t_start = time.time()
        for seedval in range(inseed,finseed):
            results += " --- Run <"+str(runindex)+"> --- seed: " + str(seedval) + " --- \n"
            print("----------------- RUN: " + str(runindex) + "/" + str(runlength) + " - SEED: " + str(seedval) + "----------------")
            self.do_seed(seedval)
            self.do_new(netmodel,verbose=0)
            history = train(model,bsize,shuffle,epochs,dset,vset)
            results += "Training time: " + str(secToTime(training_time)) + "\n"
            avtimetr += training_time
            print("Training time: " + str(secToTime(training_time)))
            x, values = test(model,tset,savequery=0)
            results += x + "\n"
            testresvalues.append(values)
            runindex += 1
        print("\n -----------------END---------------- \n")
        results += " -----------------END---------------- \n"
        avtimetr = avtimetr / runlength
        acc, f1, tpr, tnr, fnr, fpr = 0,0,0,0,0,0
        for rval in testresvalues:
            _acc, _f1, _tpr, _tnr, _fnr, _fpr, = rval
            acc += _acc
            f1 += _f1
            tpr += _tpr
            tnr += _tnr
            fnr += _fnr
            fpr += _fpr
        acc /= len(testresvalues)
        f1 /= len(testresvalues)
        tpr /= len(testresvalues)
        tnr /= len(testresvalues)
        fnr /= len(testresvalues)
        fpr /= len(testresvalues)
        t_end = time.time()
        results += "Total time: " + str(secToTime(t_end-t_start)) + "\n"
        results += "Mean values: \n"
        results += "Training time: " + str(secToTime(avtimetr)) + " \n"
        _res = "Accuracy: " + str(acc*100) + "%\n" +\
        "F1 score: " + str(f1) + "\n" +\
        "\n" +\
        "TNR: " + str(tnr) + "\n" +\
        "TPR: " + str(tpr) + "\n" +\
        "FNR: " + str(fnr) + "\n" +\
        "FPR: " + str(fpr) + "\n" 
        results += _res
        print("\n\n",results)
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d_%b_%Y(%H:%M:%S.%f)")
        filename = "RUN_"+mname+"_"+timestampStr+".log"
        with open("RUN/RESULT/"+filename, "x") as text_file:
            text_file.write(getSummary()+"\n"+results)
            print("Results saved as: ", filename)
    def do_quit(self,line):
        sys.exit()
    def do_exit(self,line):
        self.do_quit()
    def do_version(self,line):
        print("Version: " + VERSION)
    def default(self, line):
        print("Unknow command: ", line,"\nType \'help\' for commands")
        commands.append(line)

def main():
    os.system("clear")
    setSeed(rseed)
    parser = CmdParse()
    if len(sys.argv)>1:
        mname = sys.argv[1]
        parser.do_new(mname,verbose=0)
    parser.cmdloop(intro="CMDNET " + VERSION + " by Andrea Vaiuso")

if __name__ == "__main__":
    main()