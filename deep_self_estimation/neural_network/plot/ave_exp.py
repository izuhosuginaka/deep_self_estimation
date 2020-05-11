#coding:utf-8
#import wave
import numpy as np
import scipy.fftpack
from pylab import *

import sys

exp = int(sys.argv[1])

#MAX_STEP = 300
MAX_STEP = 1000

SEQ = 50

if __name__ == "__main__" :
    #1~10 ave
    f1 = open("./data_ave/exp"+str(exp)+"_table_ave.txt","w")
    #1~10
    f2 = open("./data_ave/exp"+str(exp)+"_all_ave.txt","w")
    

    data_test_plt = []
    for i in range(6):
        data_test = np.loadtxt("../data_exp/"+"exp"+str(exp)+"-test-sqrt-all-"+str(i)+".txt", delimiter=" ")
        #data_test_time = data_test[:,0]
        data_test_plt.append(data_test[:,1:SEQ+1][MAX_STEP-1])

    data_test_plt = np.asarray(data_test_plt)
    
    #print (len(data_test_plt))
    #print (len(data_test_plt[0]))
    #print (data_test_plt)
    data_test_plt = data_test_plt.mean(0)
    #print (len(data_test_plt))
    print (data_test_plt)

    
    

    for i in range(SEQ):
        f2.write("%d " %(i+1))
        f2.write("%g" %(data_test_plt[i]))
        f2.write("\n")
        

    f2.close()



    #table
    for i in range(SEQ):
        f1.write("%d " %(i+1))

    f1.write("ave\n")

    for i in range(SEQ):
        f1.write("%g " %(data_test_plt[i]))
        


    data_test_plt = []
    for i in range(6):
        data_test = np.loadtxt("../data_exp/"+"exp"+str(exp)+"-test-sqrt-ave-"+str(i)+".txt", delimiter=" ")
        #data_test_time = data_test[:,0]
        data_test_plt.append(data_test[:,1][MAX_STEP-1])

    data_test_plt = np.asarray(data_test_plt)
    data_test_plt = data_test_plt.mean()

    f1.write("%g\n" %(data_test_plt))
    
    f1.close()
    
    
