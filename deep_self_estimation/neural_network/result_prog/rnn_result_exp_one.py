#coding:utf-8
import numpy as np
import scipy.fftpack
from pylab import *
import sys
import matplotlib.pyplot as plt
import cv2

import os.path

if __name__ == "__main__" :
    
    exp = int(sys.argv[1])
    
    #toexp = sys.argv[2]

    MODEL_STEP = 1000
    
    PSIZE = 4

    TEXT_SIZE = 11

    LABEL_SIZE = 12
    
    if os.path.exists("./result_img/result_img_exp"+str(exp)):
        print ("file exists")
    else:
        print ("file does not exist")
        print ("make dir")
        os.mkdir("./result_img/result_img_exp"+str(exp))
    
    
    
    exp_num = 0
    #data_num = 18
    SEQ = 10
    
    #data = np.loadtxt("../data_result/"+"test"+"_exp"+str(exp)+"-"+str(exp_num)+".txt", delimiter=" ")
    #data = np.loadtxt("./record_result_data/"+"test"+"_exp"+str(exp)+"_"+toexp+"_"+str(exp_num)+"_model_"+str(MODEL_STEP-1)+".txt", delimiter=" ")

    data = np.loadtxt("../data_result/"+"test"+"_exp"+str(exp)+"-"+str(exp_num)+".txt", delimiter=" ")
    
    
    print (len(data))
    print (len(data[0]))

    #sys.exit()
    
    #estimation_x,estimation_y,ans_x,ans_y
    
    #est  = []
    #ans = []
    #print data[0,:2]
    #sys.exit()
    #est = np.hsplit(data, 2)[0]
    #ans = np.hsplit(data,2)[1]

    est = data[:,0:2]
    ans = data[:,2:]
    #print ("est.shape",est.shape)
    #print (est)
    #print ("ans.shape",ans.shape)
    #print (ans)
    #sys.exit()
    
    #plt.ion()
    plt.figure(figsize=(4, 4))
    fig = plt.gcf()
    
    fig.canvas.set_window_title('Graph')
    fig.patch.set_facecolor('white')
    
    
    for data_num in range(200):
        
        for i in range(SEQ):
            #赤い枠
            x = np.linspace(-20,20,4)
            y = 0*x+20.0
            plt.plot(x,y,"r-")
            x = np.linspace(-20,20,4)
            y = 0*x-20.0
            plt.plot(x,y,"r-")
            
            y = np.linspace(-20,20,4)
            x = 0*y+20.0
            plt.plot(x,y,"r-")
            y = np.linspace(-20,20,4)
            x = 0*y-20.0
            plt.plot(x,y,"r-")
            
            
            
            plt.axis([-24,24, -24, 24])
            
            #plt.legend(bbox_to_anchor=(0.5, -0.4), loc='center', borderaxespad=0,fontsize=20)
            #plt.subplots_adjust(bottom=0.5)
            plt.grid()
            
            plt.tick_params(labelsize=LABEL_SIZE)

            
            plt.plot(est[data_num*SEQ+i,0], est[data_num*SEQ+i,1],'o',label="estimation",color="b",markersize=PSIZE)
            plt.text(est[data_num*SEQ+i,0]+0.8,est[data_num*SEQ+i,1],str(i+1),fontsize=TEXT_SIZE)
            plt.plot(ans[data_num*SEQ+i,0], ans[data_num*SEQ+i,1],'o',label="Correct answer",color="r",markersize=PSIZE)
            plt.text(ans[data_num*SEQ+i,0]-2.2,ans[data_num*SEQ+i,1],str(i+1),fontsize=TEXT_SIZE)
        
            #print ("estimation",est[data_num*SEQ+i,0],est[data_num*SEQ+i,1],"answer",ans[data_num*SEQ+i,0],ans[data_num*SEQ+i,1])
            
            
            #plt.show()
            
            plt.savefig("./result_img/result_img_exp"+str(exp)+"/result_img_exp"+str(exp)+"_"+str(exp_num)+"_"+str(data_num)+"_"+str(i)+".png")
            
            plt.clf()

    
    #sys.exit()
    



    



    
