VERSION = "1.0.3.2"

from utilities import isBoolean, listFileInDir, summarizeAccuracy, summarizeLoss, qry, toBool, imgpad, secToTime
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

dsname = "None"
history = -1
bsize = 64
rseed = 49
epochs = 11
shuffle = 1
model = None
model_trained = 0
mname = "None"
training_time = 0

dlr = 1
everyepoch = 2
lrf = 0.75

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,restore_best_weights=True)
time_callback = TimeHistory()
dynamicTrain = DynamicLR(everyepoch,lrf)
reshaper = Reshaper("skip",cv2.INTER_CUBIC)

def printErr(*str):
    print(bcolors.FAIL + "".join(str) + bcolors.ENDC)


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
        printErr("File not found at: ", os.path.join(ABSPATH,pathToFile))
        return 0
    with open(os.path.join(ABSPATH,pathToFile),"r") as file:
        lines = file.readlines()
    if len(lines) != 4:
        printErr("Error on reading dataset specified on file: ", pathToFile)
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

def getDataset(pathToFile,basepath,verbose = 1,overwrite=0):
    global reshaper
    loaded_index = 0
    loaded_total = 0
    file_skipped = 0
    file_converted = 0
    lines = []
    if not os.path.isfile(os.path.join(ABSPATH,pathToFile)):
        printErr("File not found at: ", os.path.join(ABSPATH,pathToFile))
        return 0
    with open(os.path.join(ABSPATH,pathToFile),"r") as file:
        lines = file.readlines()
    loaded_total = len(lines)
    dset = Dataset([],[])
    if verbose: print("Loading data set defined in: ", pathToFile)
    if verbose: print("Total file count: ", loaded_total)
    loaded_index = 0
    postfix = []
    for line in lines:
        x = line.split()
        pth = os.path.join(basepath,x[0])
        imgname = x[0].split("/")[-1]
        img_array = cv2.imread(pth)
        if img_array is None:
            postfix.append("Cannot open image: " + os.path.abspath(pth))
            file_skipped += 1
            continue
        if img_array.shape != (ROW,COL,CH):
            if reshaper.mod == "padding":
                img_array = imgpad(img_array,ROW,COL)
                cv2.imshow("img",img_array)
                postfix.append("Reshaping image: " + imgname)
                file_converted += 1
                if overwrite: cv2.imwrite(pth,img_array)
            elif reshaper.mod == "interpolate":
                img_array = cv2.resize(img_array,(ROW,COL), reshaper.val)
                postfix.append("Reshaping image: " + imgname)
                file_converted += 1
                if overwrite: cv2.imwrite(pth,img_array)
            else:
                postfix.append("Skipping image: " + imgname)
                file_skipped += 1
                continue
        dset.data.append(img_array)
        dset.label.append(int(x[1]))
        loaded_index += 1
        suf = "loaded: " + str(loaded_index-file_skipped) + "/" + str(loaded_total) + " - skipped: " + str(file_skipped) + " - converted: " + str(file_converted)
        if verbose: printProgressBar(loaded_index,loaded_total,suffix=suf,length=30)
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
    if verbose: 
        print("Conversion completed")
        print("Data shape: ", dset.data.shape)
        print("Labels shape: ", dset.label.shape)
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
    #rates
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
    dlrstr = "lr * " + str(lrf) + " every " + str(everyepoch) + " epoch(s)" if dlr else "No"
    resh = ""
    if reshaper.mod == "skip": resh ="No"
    elif reshaper.mod == "padding": resh ="Padding"
    elif reshaper.mod == "interpolate": resh="Interpolation: " + interpmod[reshaper.val]
    smry = "Dataset selected: "+ str(dsname) +"\n"+\
        "Network: " + str(mname) + "\n" +\
        "Network trained: " + str(tr) + "\n" +\
        "Batch size: " + str(bsize) + "\n" +\
        "Epoch(s): " + str(epochs) + "\n" +\
        "Random seed: " + str(rseed) + "\n" +\
        "Shuffle: " + str(shuf) + "\n" +\
        "Reshape image: " + str(resh) + "\n" +\
        "Dynamic learning rate: " + dlrstr + "\n" +\
        "Image shape requested: (" + str(ROW) + "x" + str(COL) + "x" + str(CH) + ")\n"
    return smry

import cmd
commands = []

class CmdParse(cmd.Cmd):
    prompt = bcolors.BOLD + bcolors.OKCYAN + " (cmdnet) " + bcolors.OKGREEN + ENV_NAME + " >> " + bcolors.ENDC + "\033[0m"
    
    def do_listall(self, line):
        print(commands)
    def do_summary(self, line):
        printBorder(getSummary())
    def do_dataset(self,dset_selected):
        global dsname
        if dset_selected:
            dsetlist = listFileInDir("./DATASET",".ds",prnt=0)
            if isInt(dset_selected):
                if (int(dset_selected)-1)>=len(dsetlist) or int(dset_selected)<=0:
                    print("Available dataset: ")
                    listFileInDir("./DATASET",".ds",prnt=1)
                    return
                ds_name = dsetlist[int(dset_selected)-1]
                if os.path.isfile("./DATASET/"+ds_name):
                    getModel("./DATASET/"+ds_name,ds_name)
                else: 
                    printErr("Cannot open file: ", ds_name)
            else:
                check = setDataset("DATASET/"+dset_selected)
                if not check: printErr("Unable to load dataset: ", "./DATASET",dset_selected)
                else:
                    dsname=dset_selected
                    print("Dataset selected: ", "./DATASET/"+dset_selected)
        else:
            print("Dataset selected: ", dsname)
            print("Available datasets: ")
            listFileInDir("./DATASET",".ds",prnt=1)
    def do_ds(self,dset_selected):
        self.do_dataset(dset_selected)
    def do_batch(self,batch):
        global bsize
        if batch:
            if isInt(batch):
                bsize = int(batch)
                print("Batch size changed to: ", bsize)
            else:
                printErr("Error on parsing value as integer: ", batch)
        else: print("Batch size: ", bsize)
    def do_seed(self,sd):
        global rseed
        if sd:
            if isInt(sd):
                rseed = int(sd)
                printErr("Batch size changed to: ", rseed)
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
                printErr("Error on parsing value as integer: ", ep)
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
                printErr("Error on parsing value as boolean: ", line)
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
                printErr("Error on parsing value as boolean: ", shuf)
        else:
            if shuffle: print("Shuffle: true")
            else: print("Shuffle: false")
    def do_reshape(self,args):
        global reshaper
        params = args.split()
        if len(params) == 0:
            resh = ""
            if reshaper.mod == "skip": resh ="No"
            elif reshaper.mod == "padding": resh ="Padding"
            elif reshaper.mod == "interpolate": resh="Interpolation: " + interpmod[reshaper.val]
            print("Reshape policy: " + resh)
        else:
            if params[0].lower() == "s" or params[0].lower() == "skip":
                reshaper.mod = "skip"
                print("Reshape policy changed to: skip")
                return
            if params[0].lower() == "i" or params[0].lower() == "interpolate":
                if len(params) == 1:
                    print("Wrong value for reshape policy. You must specify one of the following interpolation algorithms:")
                    print("linear")
                    print("cubic")
                    print("nearest")
                    print("lanczos")
                    return
                if params[1].lower() == "linear":
                    reshaper.mod = "interpolate"
                    reshaper.val = cv2.INTER_LINEAR
                    print("Reshape policy changed to: interpolate linear")
                elif params[1].lower() == "cubic":
                    reshaper.mod = "interpolate"
                    reshaper.val = cv2.INTER_CUBIC
                    print("Reshape policy changed to: interpolate cubic")
                elif params[1].lower() == "nearest":
                    reshaper.mod = "interpolate"
                    reshaper.val = cv2.INTER_NEAREST
                    print("Reshape policy changed to: interpolate nearest")   
                elif params[1].lower() == "lanczos":
                    reshaper.mod = "interpolate"
                    reshaper.val = cv2.INTER_LANCZOS4
                    print("Reshape policy changed to: interpolate lanczos")
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
            else:
                print("Wrong value for reshape policy. Use 's' for skip, 'i' for interpolate or 'p' for padding")
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
                printErr("Error on parsing value as a float: ", vspl)
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
                printErr("Error while loading test set: ",testfile)
                return
            test(model,testSet)
        else:
            if qry(LABELS_TESTSET + " will be used. It\'s ok?"):
                testSet = getDataset(LABELS_TESTSET,BASEPATH)
                if testSet == 0 or testSet == -1: 
                    printErr("Error while loading test set: ",testfile)
                    return
                test(model,testSet)
            else: return
    def do_dlr(self,lr):
        global dlr
        if lr:
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
                printErr("Error on parsing value as boolean: ", lr)
        else:
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
    def do_load(self, fname):
        global ROW, COL, CH, model, mname, model_trained, bsize, epochs
        if fname:
            modellist = listFileInDir("./NETWORK",".keras",prnt=0)
            path = "./NETWORK/"+fname
            if(os.path.isfile(path)):
                getModel(path,fname)
            elif isInt(fname):
                if (int(fname)-1)>=len(modellist) or int(fname)<=0:
                    print("Available models: ")
                    listFileInDir("./NETWORK",".keras",prnt=1)
                    return
                mod_name = modellist[int(fname)-1]
                if os.path.isfile("./NETWORK/"+mod_name):
                    getModel("./NETWORK/"+mod_name,mod_name)
                    print(model.input_shape)
                else: 
                    printErr("Cannot open file: ", mod_name)
                    return
            else: 
                printErr("Cannot open file: ", path)
                return
        else:
            print("Available models: ")
            listFileInDir("./NETWORK",".keras",prnt=1)
    def do_new(self, modname):
        global model, model_trained, mname, ROW, COL, CH
        if modname:
            if modname == "malexnet":
                model = model_mAlexNet()
                ROW, COL, CH = 150, 150, 3
            elif modname == "alexnet":
                model = model_AlexNet()
                ROW, COL, CH = 224, 224, 3
            elif modname == "lenet":
                model = model_leNet()
                ROW, COL, CH = 32, 32, 1
            elif modname == "vgg16":
                model = model_VGG16()
                ROW, COL, CH = 224, 224, 3
            else:
                print("Please specify network model from listed below:\nmalexnet\nalexnet\nlenet\nvgg16")
                return
            model_trained = 0
            mname = str(model.name+"-"+modname)
        else: print("Please specify network model from listed below:\nmalexnet\nalexnet\nlenet\nvgg16")
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
        global model, bsize, shuffle, epochs, history
        if line:
            pth = os.path.join(ABSPATH,"RUN/",line)
            if not os.path.isfile(pth):
                printErr("Cannot open .run file: ", pth)
                return
            with open(pth,"r") as file:
                lines = file.readlines()
            if len(lines) != 5:
                printErr("File: ", pth," is not well formatted")
                return
            
            inseed = int(lines[0].rstrip("\n"))
            finseed = int(lines[1].rstrip("\n"))
            netmodel = lines[2].rstrip("\n")
            dstrain = lines[3].rstrip("\n")
            dstest = lines[4].rstrip("\n")
            runlength = finseed - inseed
            
            shuf = "Yes" if shuffle else "No"
            resh = ""
            if reshaper.mod == "skip": resh ="No"
            elif reshaper.mod == "padding": resh ="Padding"
            elif reshaper.mod == "interpolate": resh="Interpolation: " + interpmod[reshaper.val]
            
            smry = "RUN(s) SIZE: " + str(runlength) + "\n"
            smry += getSummary()
            printBorder(smry)
            if not qry("Proceed with run? "):
                return

            self.do_new(netmodel)
            self.do_dataset(dstrain)
            dset = getDataset(LABELS_DATA,BASEPATH)
            vset = getDataset(LABELS_VALID,BASEPATH)
            self.do_dataset(dstest)
            tset = getDataset(LABELS_TESTSET,BASEPATH)

            results = 
             + "\n"
            results += "DATASET: " + LABELS_DATA + "\n"
            results += "VALIDATION SET: " + LABELS_VALID  + "\n"
            results += "TEST SET: " + LABELS_TESTSET  + "\n\n"
            runindex = 1
            testresvalues = []
            for seedval in range(inseed,finseed):
                results += " --- Run <"+str(runindex)+"> --- seed: " + str(seedval) + " --- \n"
                print("----------------- RUN: " + str(runindex) + "/" + str(runlength) + " -----------------")
                self.do_seed(seedval)
                self.do_new(netmodel)
                history = train(model,bsize,shuffle,epochs,dset,vset)
                results += "Training time: " + str(secToTime(training_time)) + "\n"
                print("Training time: " + str(secToTime(training_time)))
                print("----------------- RUN: " + str(runindex) + "/" + str(runlength) + " -----------------")
                x, values = test(model,tset,savequery=0)
                results += x + "\n"
                testresvalues.append(values)
                runindex += 1
            results += " -----------------END---------------- \n"
            acc, f1, tpr, tnr, fnr, fpr = 0,0,0,0,0,0
            for rval in testresvalues:
                _acc, _f1, _tpr, _tnr, _fnr, _fpr = rval
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
            results += "Mean values: \n"
            _res = "Accuracy: " + str(acc*100) + "%\n" +\
            "F1 score: " + str(f1) + "\n" +\
            "\n" +\
            "TNR: " + str(tnr) + "\n" +\
            "TPR: " + str(tpr) + "\n" +\
            "FNR: " + str(fnr) + "\n" +\
            "FPR: " + str(fpr) + "\n" 
            results += _res
            print(results)
            if qry("Save run results? "):
                dateTimeObj = datetime.now()
                timestampStr = dateTimeObj.strftime("%d_%b_%Y(%H_%M_%S.%f)")
                filename = "RUN_"+mname+"_"+timestampStr+".log"
                with open("RUN/RESULT/"+filename, "x") as text_file:
                    text_file.write(getSummary()+"\n"+results)
                    print("Results saved as: ", filename)
        else:
            print("Available run files: ")
            listFileInDir("./RUN",".run",prnt=1)
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
    parser.cmdloop(intro="CMDNET " + VERSION + " by Andrea Vaiuso")

if __name__ == "__main__":
    main()