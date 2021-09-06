from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import optimizers
import numpy as np
import math
import matplotlib.pyplot as plt

BSIZE = 64
EPOCHS = 5

IMG_SIZE = (128, 128, 3)
NUM_CLASSES = 2

def getDataset():
    pass

def model_mAlexNet(input_shape):
    mAlexNet = Sequential()
    # Layer 1
    mAlexNet.add(Conv2D(16, (11,11), input_shape = input_shape, strides = (4,4),  padding='same'))
    mAlexNet.add(Activation('relu'))
    mAlexNet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    # Layer 2
    mAlexNet.add(Conv2D(20, (5,5), strides = (1,1),  padding='same'))
    mAlexNet.add(Activation('relu'))
    mAlexNet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    # Layer 3
    mAlexNet.add(Conv2D(30, (3,3), strides = (1,1),  padding='same'))
    mAlexNet.add(Activation('relu'))
    mAlexNet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    # Layer 4
    mAlexNet.add(Flatten())
    mAlexNet.add(Dense(48, activation = 'relu'))
    
    # Layer 5
    mAlexNet.add(Dense(NUM_CLASSES, activation = 'softmax'))
    
    ##Funzione di ottimizzazione di discesa lungo il gradiente (SGD)
    sgd = optimizers.SGD(learning_rate=0.01, decay=5e-4, momentum=0.9, nesterov=True)
    
    mAlexNet.compile(optimizer =sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(mAlexNet.summary())
    return mAlexNet

def train(model, data, label):
    return model.fit(data,label,validation_split=0.3, batch_size=BSIZE, shuffle=True, verbose=1, epochs=EPOCHS)

#UTILITIES
def drawCurves(history):
    #Summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","test"],loc="upper left")
    plt.show()
    #Summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def main():
    data, label = getDataset()
    model = model_mAlexNet(IMG_SIZE)
    print("Model loaded succesfully")
    history = train(model,data,label)
    drawCurves(history)

if __name__ == "__main__":
    main()