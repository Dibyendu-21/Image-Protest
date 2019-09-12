# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:28:45 2019

@author: Sonu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:50:09 2019

@author: Sonu
"""

#from weighted_cross_entrpy_loss import weighted_binary_crossentropy1, weighted_binary_crossentropy2
import pandas as pd
import numpy as np
import os
import cv2
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from sklearn.cross_validation import train_test_split
from custom_metric import fmeasure
from keras.models import Model 
import keras
from itertools import  islice
import tensorflow as tf

def read(filename):

    df = pd.read_csv(filename,encoding='utf8')

    print(df.loc[df['protest']==1,'protest'])
    a=df.loc[df['protest']==1,'protest']
    sign=df.loc[df['protest']==1,'sign']
    group_20=df.loc[df['protest']==1,'group_20']
    label0=pd.concat([sign,group_20], axis=1)

    label2=df.loc[df['protest']==1,'photo':'children']
    label3=df.loc[df['protest']==1,'group_100':'shouting']  
    label1=pd.concat([label2,label3], axis=1)    
    
    imgpath = np.empty(len(a), dtype=object)
    images =  np.empty(len(a), dtype=object)
    y = np.empty(len(a), dtype=object)
    k=0

    for i, v in a.items():
        #print('index: ', i, 'value: ', v)
        #print(df.loc[i,'fname'])
        y[k]=df.loc[i,'fname']
        k=k+1
    
    for n in range(0, len(y)):
            imgpath[n] = os.path.join(r"/home/jpstud1/Desktop/img/new/Augment_Images",y[n])
            #print(y[n]) 
            images[n] = cv2.imread(imgpath[n])
            images[n] = cv2.resize(images[n], (300, 300))
            images[n] = images[n].astype('float32')
            images[n] /= 255
        
    return images,label0,label1    
        
train_data,train_target1,train_target2=read(r'/home/jpstud1/Desktop/img/new/Augment_Images/Augment_train.csv')



img_rows=300
img_cols=300
data1=[]

for img in train_data:
    data1.append(img)

def split_validation_set_with_hold_out(train, target1, target2, test_size):
    random_state = 51
    train, X_test, target1, target1_test, target2, target2_test= train_test_split(train, target1, target2, test_size=test_size, random_state=random_state)
    X_train, X_holdout, target1_train, target1_holdout, target2_train, target2_holdout = train_test_split(train, target1, target2, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, target1_train, target1_test, target1_holdout, target2_train, target2_test, target2_holdout   

 
data_train, data_test, data_holdout, target1_train, target1_test, target1_holdout, target2_train, target2_test, target2_holdout = split_validation_set_with_hold_out(data1,train_target1,train_target2, 0.2)

print('Split train: ', len(data_train))
print('Split valid: ', len(data_test))
print('Split holdout: ', len(data_holdout))    
print()



def gen():
    print('generator initiated')
    #batch_size = 32
    i=1
    while True: 
        for i in range(0,44):
            iteru = []
            itera = islice(data_train, i*271, (i+1)*271)
            itera1 = target1_train[ i*271: (i+1)*271]
            itera2 = target2_train[ i*271: (i+1)*271]
            for img in itera:
                iteru.append(img)
            yield np.array(iteru), [np.array(itera1), np.array(itera2)]


def gen1():
    print('generator initiated')
    #batch_size = 32
    i=1
    while True: 
        for i in range(0,11):
            iteru = []
            itera = islice(data_holdout, i*271, (i+1)*271)
            itera1 = target1_holdout[ i*271: (i+1)*271]
            itera2 = target2_holdout[ i*271: (i+1)*271]
            for img in itera:
                iteru.append(img)
            yield np.array(iteru), [np.array(itera1), np.array(itera2)]


      
  
batch_size = 271
#nb_classes = 11
train_batch_length = len(data_train)
holdout_batch_length = len(data_holdout)
count=train_batch_length/batch_size
count1=holdout_batch_length/batch_size

if K.image_data_format() == 'channels_first':
    input_shape1 = (3, img_rows, img_cols)
else:
    input_shape1 = (img_rows, img_cols, 3)

input_shape = Input(shape=(img_rows, img_cols,3)) 
conv_0=Conv2D(32, (3, 3))(input_shape)
batch_0=(BatchNormalization(axis=-1))(conv_0)
act_0=Activation('relu')(batch_0)
pool_0=MaxPooling2D(pool_size=(2, 2))(act_0)

conv_1=Conv2D(32, (3, 3), activation='relu')(pool_0)
batch_1=(BatchNormalization(axis=-1))(conv_1)
act_1=Activation('relu')(batch_1)
pool_1=MaxPooling2D(pool_size=(2, 2))(act_1)

conv_2=Conv2D(64, (3, 3), activation='relu')(pool_1)
batch_2=(BatchNormalization(axis=-1))(conv_2)
act_2=Activation('relu')(batch_2)
pool_2=MaxPooling2D(pool_size=(2, 2))(act_2)

flat=Flatten()(pool_2)
dense_1=Dense(64, activation='relu')(flat)
drop_1=Dropout(0.5)(dense_1)
out_1=Dense(2, activation='sigmoid')(drop_1)
out_2=Dense(8, activation='sigmoid')(drop_1)
 
def binary_crossentropy1(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


model1= Model(inputs=input_shape, outputs=[out_1,out_2])
model1.compile(loss=[binary_crossentropy1, focal_loss], optimizer='adam',metrics=[fmeasure])
tr_gen = gen()
tr_gen1 = gen1()
m_check=keras.callbacks.ModelCheckpoint(filepath='./visual_focal_loss_combined_output_NN.h5',monitor='loss', verbose=0, save_best_only=True, mode='min')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss' , factor=0.2 , patience=5 , verbose=1 , min_lr=0.0001)
history=model1.fit_generator(generator=tr_gen, steps_per_epoch=count, nb_epoch=50, validation_data=tr_gen1, validation_steps=count1, max_queue_size=2,callbacks=[m_check])
#score = model1.evaluate(np.array(data_test),[np.array(target1_test),np.array(target2_test)], verbose=0)
#print('Score: ', score)
score = model1.evaluate_generator(generator=tr_gen1, steps=count1, max_queue_size=2, verbose=0)
print('Score holdout: ', score)

def create_submission(predictions_1,predictions_20,predictions_21):
    protest = pd.DataFrame(predictions_1, columns=['protest'])
    label1 = pd.DataFrame(predictions_20,columns=['sign','group_20'])
    label2 = pd.DataFrame(predictions_21, columns=['photo','fire','police','children'\
                                        ,'group_100','flag','night','shouting'])
    pr = protest.values
    la1 = label1.values
    la2 = label2.values
    print(la1)
    print(la2)
    print(pr)
    print(pr.shape)
    pr1=pr.flatten()
    print(pr1)
    m=0
    
    for i in range(0,len(pr1)):
        if pr1[i]>0.2:
            print(pr[i])
            m=m+1
    
    print(m)

    vio = np.empty(m, dtype=object)
    in1 = np.empty(m, dtype=object)
    k1 = np.empty(m, dtype=object)
    k2 = np.empty(m, dtype=object)
    p = np.empty(m, dtype=object)
    j=0

    wt_si=0.1
    wt_ph=0.3
    wt_fi=0.8
    wt_po=0.9
    wt_ch=-0.2
    wt_gr20=0.1
    wt_gr100=0.4
    wt_fl=0.2
    wt_ni=0.6
    wt_sh=0.7
    
    
    for i in range(0,len(pr1)):
        if pr1[i]>0.2:
            k1[j]=la1[i]
            k2[j]=la2[i]
            p[j]=pr[i]
            in1[j]=i
            j=j+1
            
        
    def column(matrix, i):
        return [row[i] for row in matrix]
            
    a=column(k1, 0)
    b=column(k1, 1)
    c=column(k2, 0)
    d=column(k2, 1)
    e=column(k2, 2)
    f=column(k2, 3)
    g=column(k2, 4)
    h=column(k2, 5)
    i=column(k2, 6)
    j=column(k2, 7)

    r=pd.DataFrame(in1, columns=['fname'])
    r0=pd.DataFrame(p, columns=['protest'])
    r1=pd.DataFrame(a, columns=['sign'])      
    print(r1)     
    r2=pd.DataFrame(b, columns=['group_20'])
    r3=pd.DataFrame(c, columns=['photo'])
    r4=pd.DataFrame(d, columns=['fire'])
    r5=pd.DataFrame(e, columns=['police'])
    r6=pd.DataFrame(f, columns=['children'])
    r7=pd.DataFrame(g, columns=['group_100'])
    r8=pd.DataFrame(h, columns=['flag'])
    r9=pd.DataFrame(i, columns=['night'])
    r10=pd.DataFrame(j, columns=['shouting'])
    result=pd.concat([r,r1,r3,r4,r5,r6,r2,r7,r8,r9,r10], axis=1)
    print(result)

    z=result.loc[:,'sign':'shouting']
    print(z)

    for i in range(0, len(result)):
        si = result.loc[i,'sign']
        ph = result.loc[i,'photo']
        fi = result.loc[i,'fire']
        po = result.loc[i,'police']
        ch = result.loc[i,'children']
        gr20 = result.loc[i,'group_20']
        gr100 = result.loc[i,'group_100']
        fl = result.loc[i,'flag']
        ni = result.loc[i,'night']
        sh = result.loc[i,'shouting']
    
        vio[i] = float(si)*wt_si + float(ph)*wt_ph + float(fi)*wt_fi +float(po)*wt_po \
        + float(ch)*wt_ch + float(gr20)*wt_gr20 + float(gr100)*wt_gr100 + float(fl)*wt_fl +\
        float(ni)*wt_ni + float(sh)*wt_sh
        print(vio[i])


    r11=pd.DataFrame(vio, columns=['violence'])
    result_total=pd.concat([result, r11], axis=1)
    print(result_total)
    
    
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    sub_file = os.path.join('subm', 'submission1_' + '.csv')
    result_total.to_csv(sub_file, index=False)

data1=[]
images_test = [cv2.imread(file) for file in glob.glob(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\demo_test\*.jpg")]
for img in images_test:
    resized_image_test = cv2.resize(img, (150, 150)) 
    resized_image_test = resized_image_test.astype('float32')
    resized_image_test /= 255
    data1.append(resized_image_test)



predictions1 = model.predict(np.array(data1), batch_size=32, verbose=1)
print(predictions1)
predictions2 = model1.predict(np.array(data1), batch_size=32, verbose=1)
print(predictions2)
create_submission(predictions1,predictions2[0],predictions2[1])

