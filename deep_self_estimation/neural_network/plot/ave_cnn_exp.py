#coding:utf-8
#import wave
import numpy as np
import scipy.fftpack
from pylab import *

import sys

exp = int(sys.argv[1])

#MAX_STEP = 300
MAX_STEP = 1000


if __name__ == "__main__" :

    f1 = open("./cnn_data/cnn_exp"+str(exp)+"_ave.txt","w")

    data_test_plt = []
    for i in range(6):
        data_test = np.loadtxt("../cnn_data_exp/"+"exp"+str(exp)+"-test-sqrt-"+str(i)+".txt", delimiter=" ")
        #data_test_time = data_test[:,0]
        data_test_plt.append(data_test[:,1][MAX_STEP-1])

    data_test_plt = np.asarray(data_test_plt)
    
    #print (len(data_test_plt))
    #print (len(data_test_plt[0]))
    #print (data_test_plt)
    data_test_plt_ave = data_test_plt.mean()
    data_test_plt_sd = np.std(data_test_plt)

    f1.write("%g " %(data_test_plt_ave))
    f1.write("%g\n" %(data_test_plt_sd))
    f1.close()
    
