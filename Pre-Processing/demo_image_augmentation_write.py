# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 23:57:57 2018

@author: Sonu
"""

import cv2 
import pandas as pd
import numpy as np
import os 

df = pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\new_demo_train\train.csv',nrows=500)

print(df.loc[df['protest']==1,'protest'])
a=df.loc[df['protest']==1,'protest']
print(a)
print(len(a))
print(a.dtype)
label=df.loc[:,:]
label.drop('violence', axis=1, inplace=True)
true_label=label.loc[df['protest']==1,:]


imgpath = np.empty(len(a), dtype=object)
images =  np.empty(len(a), dtype=object)
avging =  np.empty(len(a), dtype=object)
y = np.empty(len(a), dtype=object)
z = np.empty(len(a), dtype=object)
k=0

for i, v in a.items():
    print('index: ', i, 'value: ', v)
    print(df.loc[i,'fname'])
    y[k]=df.loc[i,'fname']
    k=k+1
    
print(y)  


for n in range(0, len(y)):
        imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\demo_train",y[n])
        images[n] = cv2.imread(imgpath[n])
        images[n] = cv2.resize(images[n], (500, 500))
        avging[n] = cv2.flip(images[n],1) 
        cv2.imshow('avg',avging[n])
        cv2.waitKey(200)
        cv2.destroyAllWindows()
        cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\img\\new\\new_demo_train\dup{}'.format(y[n]), avging[n])
'''  
for img in avging[0:5]:
    cv2.imshow('Averaging',img)
    cv2.waitKey(200)
    cv2.destroyAllWindows()
    
'''
