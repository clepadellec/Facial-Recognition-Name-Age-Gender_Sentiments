# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:55:44 2021

@author: rpic
"""
import numpy as np
import cv2
import time

cap = cv2.VideoCapture("video_input/VIDEO2.mp4")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.normalize(frame, frame, alpha=0.0, beta=1.0, cv2.NORM_MINMAX)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.05)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()