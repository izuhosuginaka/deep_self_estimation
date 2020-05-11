#仕様
#真位置と推定位置をロボットと合わせて一枚ずつプロット
#そのときの一人称視点画像
#誤差グラフをプロット

#coding:utf-8
import numpy as np
import scipy.fftpack
from pylab import *
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os
import math


exp = int(sys.argv[1])
mode_out = sys.argv[2]

exp_num = 0

if mode_out != "orb" and mode_out != "view":
    print ("no")
    sys.exit()


TEST_NUM = 1200
SEQ = 10

fig = plt.figure(figsize=(6,6))
fig.patch.set_facecolor('white')

F_SIZE = 18


for num in range(TEST_NUM):
    ims = []
    
    for seq in range(SEQ):
        if mode_out == "orb":
            #plt.title('Estimation Position',fontsize=F_SIZE)
            #推定結果
            img = cv2.imread("./result_img/result_img_exp"+str(exp)+"/result_img_exp"+str(exp)+"_"+str(exp_num)+"_"+str(num)+"_"+str(seq)+".png")
            
        if mode_out == "view":
            #plt.title('Input Image',fontsize=F_SIZE)
            #一人称視点画像
            img = cv2.imread("./view_img/view_img_exp"+str(exp)+"/rnn"+str(exp)+"_"+str(num)+"_"+str(seq)+".jpg")
        
        plt.axis("off")
        im = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ims.append([im])
        #plt.show()
        #plt.clf()
    
    #print (ims)    
    ani = animation.ArtistAnimation(fig, ims, interval=500)
    ani.save("./orbit_gif/"+mode_out+"_exp"+str(exp)+"_"+str(exp_num)+"_"+str(num)+".gif", writer="imagemagick")
    #sys.exit()
    


