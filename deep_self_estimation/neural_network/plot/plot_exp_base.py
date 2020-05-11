#coding:utf-8
import wave
import numpy as np
#import scipy.fftpack
#from pylab import *

#import dircache

import matplotlib.pyplot as plt

import sys

#exp = 58
exp = int(sys.argv[1])

MODEL_STEP = 1000


if __name__ == "__main__" :
    #plt.title("Look Around")
    #plt.title("Move Around", fontsize=23, fontname='serif')    
    
    data = np.loadtxt("./data_ave/exp"+str(exp)+"_all_ave.txt",delimiter=" ")
    
    
    data_seq = data[:,0]
    data_loss = data[:,1]

    
    err = np.loadtxt("./data_sd/exp"+str(exp)+"_all_sd.txt",delimiter=" ")
    err_seq = err[:,0]
    err_loss = err[:,1]
    
    #plot(data_sheet,data_loss, linestyle='-',label="")
    plt.plot(data_seq,data_loss, marker='o',linewidth=1.8,label="RCNN")
    plt.errorbar(data_seq,data_loss,yerr=err_loss,fmt='bo',ecolor='b',linewidth=1.8)
    
    
    #cnn_base
    data_base = np.loadtxt("./cnn_data/cnn_exp"+str(exp)+"_ave.txt",delimiter=" ")
    x = np.linspace(0,12,4)
    y = 0*x+data_base[0]
    plt.plot(x,y,"r-",label="CNN",color="r")
    
    y = 0*x+data_base[0]+data_base[1]
    plt.plot(x,y,"r--",color="r")
    y = 0*x+data_base[0]-data_base[1]
    plt.plot(x,y,"r--",color="r")

    
    #fp = FontProperties(fname=fontfile, size=20)
    #plt.legend( bbox_to_anchor=(0.5, 3), loc='upper left', borderaxespad=0)

    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0,fontsize=16)
    
    #plt.subplots_adjust(right=0.67)
    
    #plot(data_test_time, data_test_plt.mean(0), linestyle='-',label="test",color="b")
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    
    plt.ylabel("Euclidean Distans Loss [m]",fontsize=18)
    #plt.xlabel("Sequence Number",fontsize=23)
    plt.xlabel("The number of inputting images in a sequence",fontsize=18)
    

    plt.xticks(range(1,11))
    #axis([0, len(data_test_time), 0, 200])
    plt.axis([0,11,0,5.5])
    
    plt.tight_layout()
    
    #plt.legend(loc='best')
    #plt.savefig("./graph/exp"+str(exp)+"_base_model"+str(MODEL_STEP-1)+".jpg")
    plt.savefig("./graph/exp"+str(exp)+"_base_model"+str(MODEL_STEP-1)+".png")
    plt.clf()
    #show()
    #sys.exit()
    
    
