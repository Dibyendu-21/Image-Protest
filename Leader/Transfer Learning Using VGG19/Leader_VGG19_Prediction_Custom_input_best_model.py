# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:42:55 2019

@author: Sonu
"""

import pandas as pd
import numpy as np
import os
import cv2
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split
from custom_metric import fmeasure
from keras import applications
from itertools import islice


def read_new1(filename):
    df = pd.read_csv(filename, nrows=2450)
    filename=df['File']
    y=df.loc[:,'Leader']   
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    
    for n in range(0, len(filename)):
            print(filename[n])
            imgpath[n] = os.path.join(r"/home/jpstud2/Desktop/Leader_new/Image",filename[n]+'.jpg')
            images[n] = cv2.imread(imgpath[n])
            images[n] = cv2.resize(images[n], (400, 400))
            images[n] = images[n].astype('float32')
            images[n] /= 255
            
    
    return images,y
    
images_tot,Leader_tot=read_new1(r"/home/jpstud2/Desktop/Leader_new/leader_write_alternate.csv")

img_rows=400
img_cols=400
data_tot=[]
#images1 =  np.empty(len(images), dtype=object)
for img in images_tot:
    data_tot.append(img)

data_tot_train1, data_tot_test, leader_train1, leader_test= train_test_split(data_tot, Leader_tot, test_size=0.2, random_state=42)
data_tot_train, data_tot_val, leader_train,leader_val= train_test_split(data_tot_train1, leader_train1, test_size=0.2, random_state=42)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
    
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_rows, img_cols, 3))

for layer in model.layers[:38]:
    layer.trainable = False
    
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)    


def gen():
    print('generator initiated')
    #batch_size = 32
    i=1
    while True: 
        for i in range(0,32):
            iteru = []
            iteru1 = []
            itera = islice(data_tot_train, i*49, (i+1)*49)
            itera1 = islice(leader_train, i*49, (i+1)*49)
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
        for i in range(0,8):
            iteru = []
            iteru1 = []
            itera = islice(data_tot_val, i*49, (i+1)*49)
            itera1 = islice(leader_val, i*49, (i+1)*49)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)

def gen2():
    print('generator initiated')
    #batch_size = 32
    i=1
    while True: 
        for i in range(0,10):
            iteru = []
            iteru1 = []
            itera = islice(data_tot_test, i*49, (i+1)*49)
            itera1 = islice(leader_test, i*49, (i+1)*49)
            for img in itera:
                iteru.append(img)
            for img in itera1:
                iteru1.append(img)
                #print("img",img)
            #print("i",i)    
            yield np.array(iteru), np.array(iteru1)            
                  
batch_size = 49
count=len(data_tot_train)/batch_size
count1=len(data_tot_val)/batch_size
count2=len(data_tot_test)/batch_size


tr_gen = gen()
tr_gen1 = gen1()
tr_gen2 =gen2()


model_final = Model(input = model.input, output = predictions)

model_final.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=[fmeasure])
m_check=keras.callbacks.ModelCheckpoint(filepath='./Leader_VGG19_Leader_Prediction_Custom_Input_Size_best_model.h5',monitor='val_loss', save_best_only=True, verbose=0)
n_check=keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', min_delta=0, mode='min')
history=model_final.fit_generator(generator=tr_gen, steps_per_epoch=count, validation_data=tr_gen1,validation_steps=count1, max_queue_size=2, nb_epoch=50, verbose=1,callbacks=[m_check])
score = model_final.evaluate_generator(generator=tr_gen2, steps=count2, max_queue_size=2, verbose=1)
print("Test loss:",score[0])
print("Test F-Measure:",score[1])

train_predictions = model_final.predict(np.array(data_tot_train), batch_size=10, verbose=1)
test_predictions = model_final.predict(np.array(data_tot_test), batch_size=10, verbose=1)