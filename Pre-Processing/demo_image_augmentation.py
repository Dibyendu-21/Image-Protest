# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:14:46 2018

@author: Sonu
"""

import cv2
img = cv2.imread(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\demo_train\train-00005.jpg')
res = cv2.blur(img,(7,7))
cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\Presentation\\blur.jpg', res)
res1 = cv2.flip(img,1) 
cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\Presentation\\vertical_flip.jpg', res1)
res2 = cv2.flip(img,0) 
cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\Presentation\\horizontal_flip.jpg', res2)
res3 = cv2.addWeighted(img, 0.5, img, 0,0)
cv2.imwrite('C:\\Users\\Sonu\\Documents\\M.TECH\\Research\\UCLA-protest\\Presentation\\contrast.jpg', res3)
