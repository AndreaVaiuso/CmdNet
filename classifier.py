import keras
import time
import cv2
import sys
import numpy as np

INTERPOLATION_MODE = cv2.INTER_CUBIC

class Dataset:
    def  __init__(self,data,label):
       self.data = data
       self.label = label

def loadModel(path:str):
    model = keras.models.load_model(path)
    print("Model ", path, " loaded succesfully")
    ROW = model.layers[0].get_input_at(0).get_shape()[1]
    COL = model.layers[0].get_input_at(0).get_shape()[2]
    CH = model.layers[0].get_input_at(0).get_shape()[3]
    shape = (ROW, COL, CH)
    return model, shape

def classify(model:keras.Model,shape:tuple,patch:list):
    (ROW,COL,_) = shape
    t_start = time.time()
    patch = cv2.resize(patch,(ROW,COL), INTERPOLATION_MODE)
    data = [patch]
    data = np.array(data)
    pred = model.predict(data)
    t_end = time.time()
    totTime = t_end-t_start
    return pred, totTime