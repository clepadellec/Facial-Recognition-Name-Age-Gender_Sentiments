# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:17:01 2021

@author: rpic
"""
mypath="C:/Users/rpic/Documents/Challenge_SISE/training"

image_path=mypath+'/'+'PIC/PIC.jpg'

import cv2

#-----Reading the image-----------------------------------------------------
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("img",img) 