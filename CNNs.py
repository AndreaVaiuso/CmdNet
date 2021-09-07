from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, AveragePooling2D, MaxPool2D
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import keras.losses
import numpy as np
import math
import os
import time

def model_mAlexNet():
    mAlexNet = Sequential()
    # Layer 1
    mAlexNet.add(Conv2D(16, (11,11), input_shape = (150,150,3), strides = (4,4),  padding='same'))
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
    mAlexNet.add(Dense(2, activation = 'softmax'))
    
    ##Funzione di ottimizzazione di discesa lungo il gradiente (SGD)
    sgd = optimizers.SGD(learning_rate=0.01, decay=5e-4, momentum=0.9, nesterov=True)
    
    mAlexNet.compile(optimizer =sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(mAlexNet.summary())
    return mAlexNet

def model_AlexNet():
    AlexNet = Sequential()
    # Layer 1
    AlexNet.add(Conv2D(filters=96, input_shape= (32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # Layer 2
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # Layer 3
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    # Layer 4
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    # Layer 5
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # Layer 6
    AlexNet.add(Flatten())
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))

    # Layer 7
    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))

    # Layer 8
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))

    # Layer 9
    AlexNet.add(Dense(2))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('softmax'))

    AlexNet.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(AlexNet.summary())
    return AlexNet

def model_leNet():
    leNet = Sequential()
    # Layer 1
    leNet.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    leNet.add(AveragePooling2D())

    # Layer 2
    leNet.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    leNet.add(AveragePooling2D())

    # Layer 3
    leNet.add(Flatten())
    leNet.add(Dense(units=120, activation='relu'))

    # Layer 4
    leNet.add(Dense(units=84, activation='relu'))

    # Layer 5
    leNet.add(Dense(units=2, activation = 'softmax'))

    leNet.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(leNet.summary())
    return leNet

def model_VGG16():
    vgg16 = Sequential()
    # Layer 1
    vgg16.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

    # Layer 2
    vgg16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    # Layer 3
    vgg16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 4
    vgg16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    # Layer 5
    vgg16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 6
    vgg16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 7
    vgg16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    # Layer 8
    vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 9
    vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 10
    vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    # Layer 11
    vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 12
    vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

    # Layer 13
    vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    # Layer 14
    vgg16.add(Flatten())
    vgg16.add(Dense(units=4096,activation="relu"))

    # Layer 15
    vgg16.add(Dense(units=4096,activation="relu"))

    # Layer 16
    vgg16.add(Dense(units=2, activation="softmax"))

    opt = Adam(learning_rate=0.001)
    vgg16.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(vgg16.summary())
    return vgg16