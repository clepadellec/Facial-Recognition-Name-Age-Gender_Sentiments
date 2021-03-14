# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:46:59 2021

@author: rpic
"""
mypath="C:/Users/rpic/Documents/Challenge_SISE/training"
import os
from os import listdir
from os.path import isfile, join,isdir
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for e in onlyfiles:
    os.mkdir(mypath+'/'+e.split('.')[0])
onlydir = [d for d in listdir(mypath) if isdir(join(mypath, d))]
print(onlydir)
