# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 23:18:22 2018

@author: Sonu
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import cv2


def read(filename):
    df = pd.read_csv(filename)

    
    
    filename=df['File']
    filename1 = np.empty(len(filename), dtype=object)
    folder = np.empty(len(filename), dtype=object)
    xmin = np.empty(len(filename), dtype=float)
    xmin1 = np.empty(len(filename), dtype=object)
    ymin = np.empty(len(filename), dtype=float)
    ymin1 = np.empty(len(filename), dtype=object)
    xmax = np.empty(len(filename), dtype=float)
    xmax1 = np.empty(len(filename), dtype=object)
    ymax = np.empty(len(filename), dtype=float)
    ymax1 = np.empty(len(filename), dtype=object)
    x_ = np.empty(len(filename), dtype=object)
    y_ = np.empty(len(filename), dtype=object)
    x_scale = np.empty(len(filename), dtype=object)
    y_scale = np.empty(len(filename), dtype=object)
    imgpath = np.empty(len(filename), dtype=object)
    images =  np.empty(len(filename), dtype=object)
    #images_new =  np.empty(len(filename), dtype=object)
     
    for i in range(0,len(filename)):
        filename1[i] = filename[i]
        folder[i] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\Leader xml",filename[i]+'.xml')
        tree = ET.parse(folder[i])
        root = tree.getroot()
        xmin[i]=root[6][4][0].text
        ymin[i]=root[6][4][1].text
        xmax[i]=root[6][4][2].text
        ymax[i]=root[6][4][3].text
        imgpath[i] = os.path.join(r"C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\Leader",filename[i]+'.jpg')
        images[i] = cv2.imread(imgpath[i])
        y_[i] = images[i].shape[0]
        x_[i] = images[i].shape[1]
        targetSize1 = 500   
        targetSize2 = 500
        x_scale[i] = targetSize1 / x_[i]
        y_scale[i] = targetSize2 / y_[i]
        #print(x_scale[i])
        #print(y_scale[i])
        images[i] = cv2.resize(images[i], (500, 500))
        xmin1[i] = int(np.round(xmin[i] * x_scale[i]))
        ymin1[i] = int(np.round(ymin[i] * y_scale[i]))
        xmax1[i] = int(np.round(xmax[i] * x_scale[i]))
        ymax1[i] = int(np.round(ymax[i] * y_scale[i]))
        #images[i] = images[i].astype('float32')
        #images[i] /= 255
        

    return xmin1,ymin1,xmax1,ymax1,images,filename1

xmin,ymin,xmax,ymax,images,fname=read(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\leader1.csv')



def coordinate(xmin,ymin,xmax,ymax,fname):
    n0=pd.DataFrame(fname, columns=['fname'])
    n1=pd.DataFrame(xmin, columns=['xmin'])
    n2=pd.DataFrame(ymin, columns=['ymin'])
    n3=pd.DataFrame(xmax, columns=['xmax'])
    n4=pd.DataFrame(ymax, columns=['ymax'])
    result_total=pd.concat([n0,n1,n2,n3,n4], axis=1)
    
    if not os.path.isdir('leader_python'):
        os.mkdir('leader_python')
    sub_file = os.path.join('leader_python', 'new_leader' + '.csv')
    result_total.to_csv(sub_file, index=False)

coordinate(xmin,ymin,xmax,ymax,fname)



