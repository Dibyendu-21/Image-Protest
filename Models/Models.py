# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 03:47:24 2018

@author: Sonu
"""

import numpy as np
import cv2 
import pandas as pd
import os

def read_new(filename):
    df = pd.read_csv(filename)
    filename=df['fname']
    y=df.loc[:,'xmin':'ymax']    
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    
    for n in range(0, len(filename)):
            imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\Leader",filename[n]+'.jpg')
            images[n] = cv2.imread(imgpath[n])
            images[n] = cv2.resize(images[n], (150, 150))
            images[n] = images[n].astype('float32')
            images[n] /= 255
        
    for img in images[0:5]:
        cv2.imshow('new',img)
        cv2.waitKey(200) 
        cv2.destroyAllWindows()

    return images,y
    
images,y=read_new(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\leader_python\new_leader.csv")

'''
model 1
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='valid', strides=(2, 2),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu',strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation= 'linear'))

'''





'''
model 2
gray = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)

kernel_size = 3
pool_size = (2, 2)
strides = (2, 2)

model = Sequential()
    
model.add(Conv2D(32,3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
    
model.add(Conv2D(64,3,3,border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))    
'''


'''
model 3
model = Sequential()
    model.add(Convolution2D(8, (5, 5),
                            border_mode='valid',
                            input_shape=(3, image_size, image_size) ) )
    model.add(Activation('relu'))

    model.add(Convolution2D(8, (5, 5)))
    model.add(Activation('relu'))

    model.add(Convolution2D(8, (5, 5)))
    model.add(Activation('relu'))

    model.add(Convolution2D(8, (5, 5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, (5, 5)))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, (5, 5)))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, (5, 5)))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, (5, 5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4))
model.add(Activation('linear'))


'''


'''
model 4

// pseudo code
x = conv2d(x, filters=32, kernel=[3,3])->batch_norm()->relu()
x = conv2d(x, filters=32, kernel=[3,3])->batch_norm()->relu()
x = conv2d(x, filters=32, kernel=[3,3])->batch_norm()->relu()
x = maxpool(x, size=[2,2], stride=[2,2])

x = conv2d(x, filters=64, kernel=[3,3])->batch_norm()->relu()
x = conv2d(x, filters=64, kernel=[3,3])->batch_norm()->relu()
x = conv2d(x, filters=64, kernel=[3,3])->batch_norm()->relu()
x = maxpool(x, size=[2,2], stride=[2,2])

x = conv2d(x, filters=128, kernel=[3,3])->batch_norm()->relu()
x = conv2d(x, filters=128, kernel=[3,3])->batch_norm()->relu()
x = conv2d(x, filters=128, kernel=[3,3])->batch_norm()->relu()
x = maxpool(x, size=[2,2], stride=[2,2])

x = dropout()->conv2d(x, filters=128, kernel=[1, 1])->batch_norm()->relu()
x = dropout()->conv2d(x, filters=32, kernel=[1, 1])->batch_norm()->relu()

y = dense(x, units=1)

'''

'''
model 5

model = Sequential()
model.add(Conv2D(8, kernel_size=(5, 5), activation='relu', padding='valid', strides=(1, 1),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation= 'linear'))

'''


'''
model 6
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='valid', strides=(1, 1),input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))



'''