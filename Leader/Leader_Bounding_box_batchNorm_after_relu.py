# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:09:35 2019

@author: Sonu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:57:13 2019

@author: Sonu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 05:39:51 2019

@author: Sonu
"""
import pandas as pd
import numpy as np
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split


def read_new(filename):
    df = pd.read_csv(filename)
    filename=df['fname']
    y=df.loc[:,'xmin':'ymax']    
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    
    for n in range(0, len(filename)):
            imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\Leader",filename[n]+'.jpg')
            images[n] = cv2.imread(imgpath[n])
            images[n] = cv2.resize(images[n], (500, 500))
            images[n] = images[n].astype('float32')
            images[n] /= 255
    
    return images,y,filename       
    
images,loc,file=read_new(r"C:\Users\Sonu\Music\Leader\new\new_leader_500.csv")


img_rows=500
img_cols=500
data=[]
i=0
#images1 =  np.empty(len(images), dtype=object)
for img in images:
    data.append(img)
    

data_tot_train, data_test, label_tot_train, label_test ,file_tot_train, file_test= train_test_split(data, loc, file, test_size=0.2, random_state=42)
data_train, data_val, label_train, label_val, file_train, file_val=train_test_split(data_tot_train, label_tot_train, file_tot_train, test_size=0.2, random_state=42)

print(file_test[0:5])
print(label_test[0:5])

for j in range(0,5):    
    cv2.imshow('img',data_test[j])
    cv2.waitKey(2000) 
    cv2.destroyAllWindows()




if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)



model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='valid', strides=(2, 2),input_shape=input_shape))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu',strides=(1, 1), padding='valid'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation= 'linear')) 

def distance_loss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)), axis=0) 


def iou_loss(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss

def iou_metric(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return iou    


model.compile(loss= distance_loss, optimizer=keras.optimizers.Adam(lr = 0.001), metrics=[iou_metric])

m_check=keras.callbacks.ModelCheckpoint(filepath='./Leader_Bounding_Box_Custom_model_original_batch_after_relu.h5',monitor='val_loss', save_best_only=True, verbose=0)
n_check=keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', min_delta=0, mode='min')
history=model.fit(np.array(data_train), np.array(label_train), batch_size=10, epochs=50, validation_data=(np.array(data_val),np.array(label_val)), verbose=1,callbacks=[m_check])
score = model.evaluate(np.array(data_test), np.array(label_test), batch_size=10, verbose=1)
print("Test loss:",score[0])
print("Test AUC:",score[1])

train_predictions = model.predict(np.array(data_train), batch_size=10, verbose=1)
test_predictions = model.predict(np.array(data_test), batch_size=10, verbose=1)

def create_newSubmission(train_fname, train_predictions):
    n0=pd.DataFrame(train_fname, columns=['fname'])
    n1=pd.DataFrame(train_predictions, columns=['xmin','ymin','xmax','ymax'])
    result_total=pd.concat([n0], axis=1)
    result_total1=pd.concat([n1], axis=1)
    if not os.path.isdir('leader_python'):
        os.mkdir('leader_python')
    sub_file = os.path.join('leader_python', 'leader_filename_train_originial_model_batch_after_relu' + '.csv')
    result_total.to_csv(sub_file, index=False)
    sub_file_1 = os.path.join('leader_python', 'leader_coordinate_train_originial_model_batch_after_relu' + '.csv')
    result_total1.to_csv(sub_file_1, index=False)
    
create_newSubmission(file_train,train_predictions)    



def create_newSubmission_test(test_fname, test_predictions):
    n0=pd.DataFrame(test_fname, columns=['fname'])
    n1=pd.DataFrame(test_predictions, columns=['xmin','ymin','xmax','ymax'])
    result_total=pd.concat([n0], axis=1)
    result_total1=pd.concat([n1], axis=1)
    if not os.path.isdir('leader_python'):
        os.mkdir('leader_python')
    sub_file = os.path.join('leader_python', 'leader_filename_test_originial_model_batch_after_relu' + '.csv')
    result_total.to_csv(sub_file, index=False)
    sub_file_1 = os.path.join('leader_python', 'leader_coordinate_test_originial_model_batch_after_relu' + '.csv')
    result_total1.to_csv(sub_file_1, index=False)
    
create_newSubmission_test(file_test,test_predictions)    