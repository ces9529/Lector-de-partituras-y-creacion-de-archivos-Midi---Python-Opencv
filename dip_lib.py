# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:48:33 2017

@author: césar
"""
import cv2
from matplotlib import pyplot as plt

def show (img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return
def load(ruta, color = True):
    img = cv2.imread(ruta)
    if(color):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
    return img