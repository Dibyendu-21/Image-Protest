# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:50:14 2018

@author: Sonu
"""

import cv2
import random
import pandas as pd
import numpy as np
import os 

df = pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Augment Images\train.csv')

print(df.loc[df['protest']==1,'protest'])
a=df.loc[df['protest']==1,'protest']
#print(a)
print(len(a))
print(a.dtype)
label=df.loc[:,:]
label.drop('violence', axis=1, inplace=True)
true_label=label.loc[df['protest']==1,:]


imgpath = np.empty(len(a), dtype=object)
images =  np.empty(len(a), dtype=object)
res =  np.empty(len(a), dtype=object)
y = np.empty(len(a), dtype=object)
z = np.empty(len(a), dtype=object)
k=0

for i, v in a.items():
    print('index: ', i, 'value: ', v)
    print(df.loc[i,'fname'])
    y[k]=df.loc[i ,'fname']
    k=k+1
    
  


for n in range(0, len(y)):
        imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\train",y[n])
        images[n] = cv2.imread(imgpath[n])
        #print(y[n])
        images[n] = cv2.resize(images[n], (500, 500))


i=0

for n in range(0, len(y)):
    x = random.randint(0, 3)
    if x == 0:
        #blur
        #cv2.imshow('original',images[n])
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows() 
        res[i] = cv2.blur(images[n],(5,5))
        #cv2.imshow('BLUR',res[i])
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        i=i+1
        cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\img\\new\\Augment Images\dup{}'.format(y[n]), res[n])
    elif x==1:
        # Vertical Flipping. 
        #cv2.imshow('original',images[n])
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        res[i] = cv2.flip(images[n],1) 
        #cv2.imshow('VERTCAL FLIP',res)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        i=i+1
        cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\img\\new\\Augment Images\dup{}'.format(y[n]), res[n])
    elif x==2:
        #Horizontal Flipping
        #cv2.imshow('original',images[n])
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        res[i] = cv2.flip(images[n],0) 
        #cv2.imshow('HORIZONTAL FLIP',res)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows() 
        i=i+1
        cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\img\\new\\Augment Images\dup{}'.format(y[n]), res[n])
    else:
        #Contrast Night shade
        #cv2.imshow('original',images[n])
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        res[i] = cv2.addWeighted(images[n], 0.5, images[n], 0,0)
        #cv2.imshow('CONTRAST',res)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows() 
        i=i+1
        cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\img\\new\\Augment Images\dup{}'.format(y[n]), res[n])