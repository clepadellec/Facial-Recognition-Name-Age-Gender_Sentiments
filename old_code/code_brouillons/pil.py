# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:33:12 2021

@author: rpic
"""
import cv2
import face_recognition
import os
import pickle
from PIL import Image, ImageEnhance


image = Image.open('C:/Users/rpic/Documents/Challenge_SISE/brightness.jpg')

sharpness = ImageEnhance.Sharpness(image)
sharpness.enhance(1.5).save('sharpness.jpg')