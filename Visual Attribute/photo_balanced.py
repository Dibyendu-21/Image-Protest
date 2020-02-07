# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:27:10 2019

@author: Sonu
"""

import pandas as pd
import numpy as np
import os
import cv2
#import tensorflow as tf
import keras
#import keras.backend.tensorflow_backend as tfb
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Activation, Flatten, Dense
from keras import backend as K
from sklearn.cross_validation import train_test_split
from custom_metric import roc_auc_score,fmeasure
#from keras.models import Model 
from itertools import islice
from keras.layers.normalization import BatchNormalization

def read(filename):
    df = pd.read_csv(filename)
    print(df.head())
    label = df['photo']
    print(label.value_counts())
    filename=df['fname']
    
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    #server path - /home/jpstud1/Desktop/img/new/new_Augment_Images
    for n in range(0, len(filename)):
        imgpath[n] = os.path.join(r"/home/jpstud1/Desktop/img/new/new_augment",filename[n])
        images[n] = cv2.imread(imgpath[n])
        images[n] = cv2.resize(images[n], (400, 400))
        images[n] = images[n].astype('float32')
        images[n] /= 255
        #data.append(images[n])
        
    
    return images,label
    
train_data,train_target=read(r'/home/jpstud1/Desktop/img/new/new_augment/photo_balanced.csv')
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
        for i in range(0,25):
            iteru = []
            iteru1 = []
            itera = islice(data_train, i*92, (i+1)*92)
            itera1 = islice(label_train, i*92, (i+1)*92)
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
        for i in range(0,8):
            iteru = []
            iteru1 = []
            itera = islice(data_holdout, i*72, (i+1)*72)
            itera1 = islice(label_holdout, i*72, (i+1)*72)
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
        for i in range(0,10):
            iteru = []
            iteru1 = []
            itera = islice(data_test, i*72, (i+1)*72)
            itera1 = islice(label_test, i*72, (i+1)*72)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)
  
batch_size_train = 92
batch_size_test = 72
#nb_classes = 11
train_batch_length = len(label_train)
holdout_batch_length= len(label_holdout)
test_batch_length = len(label_test)
count=train_batch_length/batch_size_train
count1=holdout_batch_length/batch_size_test
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
model.add(Dropout(0.2))

model.add(Conv2D(8, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(16, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[fmeasure,roc_auc_score])


tr_gen = gen()
tr_gen1 = gen1()
tr_gen2 = gen2()
m_check=keras.callbacks.ModelCheckpoint(filepath='./photo_augment500.h5',monitor='val_loss', verbose=0, save_best_only=True, mode='min')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0001 )
history=model.fit_generator(generator=tr_gen, steps_per_epoch=count, nb_epoch=50, validation_data=tr_gen1, validation_steps=count1, max_queue_size=2,callbacks=[m_check])

#model.fit(np.array(data_train),np.array(label_train), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(np.array(data_test),np.array(label_test)))
score = model.evaluate_generator(generator=tr_gen2,steps=count2,max_queue_size=2, verbose=0)
print('Score: ', score)
score = model.evaluate_generator(generator=tr_gen1,steps=count1,max_queue_size=2, verbose=0)
print('Score holdout: ', score)

df1=pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Violence_test\violence_test.csv')
fname=df1['fname']
imgpath=np.empty(len(fname), dtype=object)
images=np.empty(len(fname), dtype=object)
for n in range(0, len(fname)):
        imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Violence_test",fname[n])
        images[n] = cv2.imread(imgpath[n])
        images[n] = cv2.resize(images[n], (400, 400))
        images[n] = images[n].astype('float32')
        images[n] /= 255
        
        
predicted_score=model.predict(np.array(images), verbose=0)        

augment1 = pd.DataFrame(score, columns=['preict_score'])         
if not os.path.isdir('Predicted_score'):
    os.mkdir('Predicted_score')
sub_file1 = os.path.join('Predicted_score', 'violence_test' + '.csv')
#sub_file2 = os.path.join('Image_Augmentation_new', 'firedup1_value' + '.csv')
augment1.to_csv(sub_file1, index=False)
      