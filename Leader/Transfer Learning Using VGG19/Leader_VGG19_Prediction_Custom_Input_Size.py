# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 06:47:18 2019

@author: Sonu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:28:08 2019

@author: Sonu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 00:51:31 2018

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

def read_new1(filename):
    df = pd.read_csv(filename, nrows=915)
    filename=df['File']
    y=df.loc[:,'Leader']   
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    
    for n in range(0, len(filename)):
            imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\Leader",filename[n]+'.jpg')
            images[n] = cv2.imread(imgpath[n])
            images[n] = cv2.resize(images[n], (400, 400))
            images[n] = images[n].astype('float32')
            images[n] /= 255
    
    
    return images,y
    
images_tot,Leader_tot=read_new1(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\leader_write_alternate.csv")

img_rows=400
img_cols=400
data_tot=[]
#images1 =  np.empty(len(images), dtype=object)
for img in images_tot:
    data_tot.append(img)

data_tot_train, data_tot_test, leader_train, leader_test= train_test_split(data_tot, Leader_tot, test_size=0.2, random_state=42)


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


model_final = Model(input = model.input, output = predictions)

model_final.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['fmeasure'])
m_check=keras.callbacks.ModelCheckpoint(filepath='./Leader_VGG19_Leader_Prediction_Custom_Input_Size.h5', verbose=0)


history=model_final.fit(np.array(data_tot_train), np.array(leader_train), batch_size=10, epochs=10, verbose=1,callbacks=[m_check])
score = model_final.evaluate(np.array(data_tot_test), np.array(leader_test), batch_size=10, verbose=1)
print("Test loss:",score[0])
print("Test F-Measure:",score[1])

train_predictions = model_final.predict(np.array(data_tot_train), batch_size=10, verbose=1)
test_predictions = model_final.predict(np.array(data_tot_test), batch_size=10, verbose=1)










