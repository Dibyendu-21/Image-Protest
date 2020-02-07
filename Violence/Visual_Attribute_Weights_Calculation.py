# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:07:45 2019

@author: Sonu
"""

import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\trainer.csv',encoding='utf8')
att=df.loc[df['protest']==1,'sign':'shouting']
att.astype('int64')
x= np.empty(len(att),dtype=int)
print(x.dtype)
x=np.array(att)
print(x.shape)
for j in range(0,len(x)):
    for i in range(0,10):
        x[j][i]=float(x[j][i])        
print(x.dtype)
vio=df.loc[df['protest']==1,'violence']
y=np.array(vio)
for k in range(0,len(vio)):
        y[k]=float(y[k])  
mean_vio=np.std(y)
print(mean_vio)
print(y.shape)
new= np.empty(len(vio),dtype=float)
out= np.empty(10,dtype=int)
out=np.linalg.lstsq(x, y)[0]
print(out)
new=np.dot(x,out)
for i in range(0,5):
    print(new[i])
print(new.shape)
mean_new=np.std(new)
print(mean_new)

