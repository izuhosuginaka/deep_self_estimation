#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import mcpi.minecraft as minecraft
#import pyautogui
import time
import math
import numpy as np

from numpy.random import *

#import sympy.geometry as sg
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import os.path
import cv2

exp = int(sys.argv[1])

if os.path.exists("./view_img/view_img_exp"+str(exp)):
  print ("file exists")
else:
  print ("file does not exist")
  print ("make dir")
  os.mkdir("./view_img/view_img_exp"+str(exp))

n_input_row = 48
n_input_col = 48
  
DATA_MAX = 1200
SEQ = 10


for i in range(DATA_MAX):  
  for seq in range(SEQ):
    
    #一人称視点画像保存
    img = cv2.imread("../../screenshot/scsho/rnn"+str(exp)+"/rnn"+str(exp)+"_"+str(i)+"_"+str(seq)+".jpg")
    img = img[:,400-240:400+240]
    img = cv2.resize(img, (n_input_row, n_input_col))
    cv2.imwrite("./view_img/view_img_exp"+str(exp)+"/rnn"+str(exp)+"_"+str(i)+"_"+str(seq)+".jpg",img)
    
