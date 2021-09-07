from CNNs import model_mAlexNet
from CNNs import model_AlexNet
from utilities import drawCurves

import numpy as np
import math

BSIZE = 64
EPOCHS = 5

def getDataset():
    pass

def train(model, data, label):
    return model.fit(data,label,validation_split=0.3, batch_size=BSIZE, shuffle=True, verbose=1, epochs=EPOCHS)

def testMAlexNet():
    data, label = getDataset()
    model = model_mAlexNet() # (150x150x3)
    print("Model loaded succesfully")
    history = train(model,data,label)
    drawCurves(history)

def testAlexNet():
    data, label = getDataset()
    model = model_AlexNet() # (32x32x3)
    print("Model loaded succesfully")
    history = train(model,data,label)
    drawCurves(history)

def main():
    testAlexNet()

if __name__ == "__main__":
    main()