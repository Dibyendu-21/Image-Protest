# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:35:00 2018

@author: Sonu
"""
import glob
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from sklearn.cross_validation import train_test_split
from custom_metric import mcc,roc_auc_score,fmeasure
from keras.models import Model 
from itertools import chain, islice

def read(filename):
    df = pd.read_csv(filename,encoding='utf8', nrows=32600)

    df = df.replace('-',np.NaN)
    
    df = df.fillna(0)
    print(df.head())
    label = df['protest']
    print(label.value_counts())
    filename=df['fname']
    
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    
    for n in range(0, len(filename)):
        imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\train",filename[n])
        images[n] = cv2.imread(imgpath[n])
        images[n] = cv2.resize(images[n], (150, 150))
        images[n] = images[n].astype('float32')
        images[n] /= 255
        #data.append(images[n])
        
    
    return images,label
    
train_data,train_target=read(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Augment Images\train.csv')


img_rows=150
img_cols=150
data=[]

for img in train_data:
    data.append(img)

def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout    

 
data_train, data_test, data_holdout, label_train, label_test, label_holdout = split_validation_set_with_hold_out(data,train_target, 0.2)



def gen():
    print('generator initiated')
    #batch_size = 32
    i=1
    while True: 
        for i in range(0,64):
            iteru = []
            iteru1 = []
            itera = islice(data_train, i*326, (i+1)*326)
            itera1 = islice(label_train, i*326, (i+1)*326)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)


def gen1():
    print('generator initiated')
    #batch_size = 32
    i=1
    while True: 
        for i in range(0,20):
            iteru = []
            iteru1 = []
            itera = islice(data_test, i*326, (i+1)*326)
            itera1 = islice(label_test, i*326, (i+1)*326)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)
      
  
batch_size = 326
#nb_classes = 11
nb_epoch = 5
train_batch_length = len(label_train)
test_batch_length = len(label_test)
count=train_batch_length/batch_size
count1=test_batch_length/batch_size

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=[fmeasure,'accuracy'])


tr_gen = gen()
tr_gen1 = gen1()
model.fit_generator(generator=tr_gen, steps_per_epoch=count, nb_epoch=50, validation_data=tr_gen1, validation_steps=count1, max_queue_size=2)

#model.fit(np.array(data_train),np.array(label_train), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(np.array(data_test),np.array(label_test)))
#score = model.evaluate(np.array(data_test),np.array(label_test), verbose=0)
#print('Score: ', score)
#score = model.evaluate(np.array(data_holdout),np.array(label_holdout), verbose=0)
#print('Score holdout: ', score)


















