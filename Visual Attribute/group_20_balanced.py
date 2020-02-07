# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:28:49 2019

@author: Sonu
"""

#from google.colab import drive
#drive.mount('/content/my_drive/')
#!ls
#!pwd
#cd my_drive
#cd My\ Drive
#cd group_20_augment
import pandas as pd
import numpy as np
import os
import cv2
#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Activation, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from custom_metric import roc_auc_score,fmeasure
#from keras.models import Model 
from itertools import islice
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
def read(filename):
    df = pd.read_csv(filename, nrows =3096)
    print(df.head())
    label = df['group_20']
    print(label.value_counts())
    filename=df['fname']
    
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    #server path - /home/jpstud1/Desktop/img/new/new_Augment_Images
    for n in range(0, len(filename)):
        imgpath[n] = filename[n]
        images[n] = cv2.imread(imgpath[n])
        images[n] = cv2.resize(images[n], (400, 400))
        images[n] = images[n].astype('float32')
        images[n] /= 255
        #data.append(images[n])
        
    
    return images,label
    
train_data,train_target=read('group_20_balanced.csv')
#csv path - /home/jpstud1/Desktop/img/new/new_Augment_Images/photo_balanced.csv

img_rows=400
img_cols=400
data=[]

for img in train_data:
    data.append(img)

def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout    

 
data_train, data_test, data_holdout, label_train, label_test, label_holdout = split_validation_set_with_hold_out(data,train_target, 0.2)


print(len(data_train))
print(len(data_test))
print(len(data_holdout))
print(len(label_train))
print(len(label_test))
print(len(label_holdout))


def gen():
    print('generator initiated')
    i=1
    while True: 
        for i in range(0,320):
            iteru = []
            iteru1 = []
            itera = islice(data_train, i*10, (i+1)*10)
            itera1 = islice(label_train, i*10, (i+1)*10)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)


def gen1():
    print('generator initiated')
    i=1
    while True: 
        for i in range(0,80):
            iteru = []
            iteru1 = []
            itera = islice(data_holdout, i*10, (i+1)*10)
            itera1 = islice(label_holdout, i*10, (i+1)*10)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)
      

def gen2():
    print('generator initiated')
    i=1
    while True: 
        for i in range(0,50):
            iteru = []
            iteru1 = []
            itera = islice(data_test, i*20, (i+1)*20)
            itera1 = islice(label_test, i*20, (i+1)*20)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)
  
batch_size_train = 10
batch_size_holdout = 10
batch_size_test = 20
#nb_classes = 11
train_batch_length = len(label_train)
holdout_batch_length= len(label_holdout)
test_batch_length = len(label_test)
count=train_batch_length/batch_size_train
count1=holdout_batch_length/batch_size_holdout
count2=test_batch_length/batch_size_test

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(8, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(10, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.45))

model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.65))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[fmeasure])


tr_gen = gen()
tr_gen1 = gen1()
tr_gen2 = gen2()
m_check=keras.callbacks.ModelCheckpoint(filepath='./group_20_augment500.h5',monitor='val_fmeasure', verbose=0, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0001 )
history=model.fit_generator(generator=tr_gen, steps_per_epoch=count, nb_epoch=80, validation_data=tr_gen1, validation_steps=count1, max_queue_size=2,callbacks=[m_check])

#model.fit(np.array(data_train),np.array(label_train), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(np.array(data_test),np.array(label_test)))
score = model.evaluate_generator(generator=tr_gen2,steps=count2,max_queue_size=2, verbose=0)
print('Score: ', score)
score = model.evaluate_generator(generator=tr_gen1,steps=count1,max_queue_size=2, verbose=0)
print('Score holdout: ', score)

print("With best model")
file_path='./group_20_augment500.h5'
model1 = load_model(file_path,custom_objects={'fmeasure': fmeasure})
score1 = model1.evaluate_generator(generator=tr_gen2,steps=count2,max_queue_size=2, verbose=0)
print('Score: ', score1)
score2 = model1.evaluate_generator(generator=tr_gen1,steps=count1,max_queue_size=2, verbose=0)
print('Score holdout: ', score2)