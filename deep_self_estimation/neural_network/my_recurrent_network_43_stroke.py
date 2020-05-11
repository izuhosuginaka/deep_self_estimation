
from __future__ import print_function

import sys
import cv2

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
#from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn.python.ops import core_rnn

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(2323)

exp = 43

TRAIN_SIZE = 1000
TEST_SIZE = 200
NUM_MAX = int((TRAIN_SIZE+TEST_SIZE)/TEST_SIZE)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '../screenshot/scsho_txt/rnn'+str(43)+'.txt', 'File name of data dir')

# Parameters
learning_rate = 0.0001

max_epoch = 1000
batch_size = 20

CALC_BATCH = 100

SAVE_MODEL_STEP = 100


n_input_row = 48 
n_input_col = 48
n_steps = 10

#CNN
n_fc1 = 1024
n_fc2 = 256

n_hidden = 128 # hidden layer num of features

n_classes = 2 # MNIST total classes (0-9 digits)

images_placeholder = tf.placeholder("float", [None, n_steps, n_input_row*n_input_col*3])
labels_placeholder = tf.placeholder("float", [None, n_steps, n_classes])


# 重みを標準偏差0.1の正規分布で初期化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアスを標準偏差0.1の正規分布で初期化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#CNN
# Define weights
weights = {
    #CNN
    'W_conv1': weight_variable([5, 5, 3, 32]),
    'W_conv2': weight_variable([5, 5, 32, 64]),
    'W_fc1': weight_variable([int(n_input_row/4*n_input_col/4*64), n_fc1]),
    'out': weight_variable([n_hidden, n_classes])    
}

biases = {
    #CNN
    'b_conv1': bias_variable([32]),
    'b_conv2': bias_variable([64]),
    'b_fc1': bias_variable([n_fc1]),
    'out': bias_variable([n_classes])    
}

def LRCN(images_placeholder, weights, biases, keep_prob1, keep_prob2, keep_prob3, keep_prob4, keep_prob5, keep_prob6, keep_prob7):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    images_placeholder = tf.transpose(images_placeholder, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    images_placeholder = tf.reshape(images_placeholder, [-1, n_input_row*n_input_col*3])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    #images_placeholder = tf.split(0, n_steps, images_placeholder)
    images_placeholder = tf.split(images_placeholder, n_steps, 0)
    
    
    #CNN
    
    # 畳み込み層の作成
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
        
    lstm_input = []
    
    for time_step in range(n_steps):
        # 入力を28x28x1に変形
        x_image = tf.reshape(images_placeholder[time_step], [-1, n_input_row, n_input_col, 3])
        
        # 畳み込み層1
        with tf.name_scope('conv1') as scope:
            x_image_drop = tf.nn.dropout(x_image, keep_prob1)
            h_conv1 = tf.nn.relu(conv2d(x_image_drop, weights['W_conv1']) + biases['b_conv1'])

        # プーリング層1
        with tf.name_scope('pool1') as scope:
            h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob2)
            h_pool1 = max_pool_2x2(h_conv1_drop)

        # 畳み込み層2
        with tf.name_scope('conv2') as scope:
            h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob3)
            h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, weights['W_conv2']) + biases['b_conv2'])

        # プーリング層2
        with tf.name_scope('pool2') as scope:
            h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob4)
            h_pool2 = max_pool_2x2(h_conv2_drop)
        
        # 全結合層1
        with tf.name_scope('fc1') as scope:
            h_pool2_flat = tf.reshape(h_pool2, [-1,int(n_input_row/4*n_input_col/4*64)])
            h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat,keep_prob5)
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat_drop, weights['W_fc1']) + biases['b_fc1'])
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob6)#

        lstm_input.append(h_fc1_drop)
        
        
    #RNN
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    #lstm_cell = rnn.BasicLSTMCell(n_hidden) 
    
    # Get lstm cell output
    #outputs, states = rnn.rnn(lstm_cell, lstm_input, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, lstm_input, dtype=tf.float32)
    
    #線形活性(time_steps分の出力)
    out = []
    for time_step in range(n_steps):
        out_uni = tf.matmul(outputs[time_step], weights['out']) + biases['out']
        out_uni_drop = tf.nn.dropout(out_uni, keep_prob7)
        out.append(out_uni_drop)
        
    out = tf.transpose(out, [1, 0, 2])    
    return out


def loss(pred, labels_placeholder):
    #error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    error = tf.reduce_mean(tf.square(pred-labels_placeholder))
    
    return error


# ファイルを開く
f = open(FLAGS.dataset, 'r')
# データを入れる配列
dataset_image = []
dataset_label = []
print("...loading dataset")
for line in f:
    # 改行を除いてスペース区切りにする
    line = line.rstrip()
    l = line.split()
    # 画像読み込み
    img = cv2.imread(l[0])
    #正方形にトリミング
    img = img[:,400-240:400+240]
    img = cv2.resize(img, (n_input_row, n_input_col))
    # 一列にした後、0-1のfloat値にする
    dataset_image.append(img.flatten().astype(np.float32)/255.0)

    x = float(l[1])
    y = float(l[3])
    z = float(l[2])
    angle = float(l[4])
    pitch = float(l[5])
    
    #クラス2:位置のみ推定
    tmp = np.zeros(n_classes)
    tmp[0] = x
    tmp[1] = y
    dataset_label.append(tmp)
    #sys.exit()
    
f.close()

dataset_image = np.asarray(dataset_image)
dataset_label = np.asarray(dataset_label)

#print (len(dataset_image))
#print (len(dataset_label))
DATA_SIZE = len(dataset_image)

if DATA_SIZE%n_steps!=0:
    print ("MY_ERROR : DATA_SIZE_SEQ")
    sys.exit()

DATA_SIZE_SEQ = int(DATA_SIZE/n_steps)
#print (DATA_SIZE_SEQ)#1200
#print (len(dataset_image))#12000
#print (len(dataset_image[0]))#6912
#sys.exit()

dataset_image = dataset_image.reshape((DATA_SIZE_SEQ,n_steps,n_input_row*n_input_col*3))
#print (len(dataset_image))#1200
#print (len(dataset_image[0]))#10
#print (len(dataset_image[0][0]))#48*48*3

dataset_label = dataset_label.reshape((DATA_SIZE_SEQ,n_steps,n_classes))
#print (len(dataset_label))#1200
#print (len(dataset_label[0]))#10
#print (len(dataset_label[0][0]))#2

dataset_image = np.asarray(dataset_image)
dataset_label = np.asarray(dataset_label)

keep_prob1 = tf.placeholder("float")
keep_prob2 = tf.placeholder("float")
keep_prob3 = tf.placeholder("float")
keep_prob4 = tf.placeholder("float")
keep_prob5 = tf.placeholder("float")
keep_prob6 = tf.placeholder("float")
keep_prob7 = tf.placeholder("float")

pred = LRCN(images_placeholder, weights, biases, keep_prob1, keep_prob2, keep_prob3, keep_prob4, keep_prob5, keep_prob6, keep_prob7)

#sys.exit()

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = loss(pred, labels_placeholder)
#cost = tf.reduce_mean(tf.square(pred-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,2), tf.argmax(y,2))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
#init = tf.initialize_all_variables()

# 保存の準備
saver = tf.train.Saver(max_to_keep = 0)

#sys.exit()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph
with tf.Session(config=config) as sess:
    
    num = 0
    while num<NUM_MAX:    
        #MODEL_PATH = "./model/model"+str(exp)+"-"+str(num)+".ckpt"
        MODEL_PATH = "./model/rnn_model"+str(exp)+"-"+str(num)+".ckpt"
    
        traindir = "./data_exp/exp"+str(exp)+"-train-"+str(num)+".txt"
        testdir = "./data_exp/exp"+str(exp)+"-test-"+str(num)+".txt"

        traindir_sqrt_all = "./data_exp/exp"+str(exp)+"-train-sqrt-all-"+str(num)+".txt"
        traindir_sqrt_ave = "./data_exp/exp"+str(exp)+"-train-sqrt-ave-"+str(num)+".txt"
        
        testdir_sqrt_all = "./data_exp/exp"+str(exp)+"-test-sqrt-all-"+str(num)+".txt"
        testdir_sqrt_ave = "./data_exp/exp"+str(exp)+"-test-sqrt-ave-"+str(num)+".txt"
        
        fp_train = open(traindir,"w")
        fp_test = open(testdir,"w")    
        
        fp_train_sqrt_all = open(traindir_sqrt_all,"w")    
        fp_train_sqrt_ave = open(traindir_sqrt_ave,"w")
        
        fp_test_sqrt_all = open(testdir_sqrt_all,"w")    
        fp_test_sqrt_ave = open(testdir_sqrt_ave,"w")
        
        #result
        result_train_dir = "./data_result/train_exp"+str(exp)+"-"+str(num)+".txt"
        result_test_dir = "./data_result/test_exp"+str(exp)+"-"+str(num)+".txt"
        
        # 変数の初期化
        sess.run(tf.initialize_all_variables())
        
        #データセットをテストデータ,訓練データ分ける
        #交差検証法
        if DATA_SIZE_SEQ%TEST_SIZE!=0:
            print ("MY_ERROR : TEST|TRAIN DATA")
            sys.exit()
    
        k = int(DATA_SIZE_SEQ/TEST_SIZE)
        train_image = np.vstack((dataset_image[:TEST_SIZE*num,],dataset_image[TEST_SIZE*(num+1):,]))
        train_label = np.vstack((dataset_label[:TEST_SIZE*num,],dataset_label[TEST_SIZE*(num+1):,]))
    
        test_image = dataset_image[TEST_SIZE*num:TEST_SIZE*(num+1),]
        test_label = dataset_label[TEST_SIZE*num:TEST_SIZE*(num+1),]
    
        print (len(train_image))
        print (len(test_image))
        #sys.exit()
        
        
        # 訓練の実行
        for step in range(max_epoch):
            print (num,"/",k,("step:{0}".format(step)))

            #シャッフル系列作成
            random_seq = np.random.permutation(len(train_image))
            train_image_batch = []
            train_label_batch = []
            for i in range(int(len(train_image)/batch_size)):
                batch = batch_size*i
                for j in range(batch_size):
                    train_image_batch.append(train_image[random_seq[batch+j]])
                    train_label_batch.append(train_label[random_seq[batch+j]])

            train_image_batch = np.asarray(train_image_batch)
            train_label_batch = np.asarray(train_label_batch)
            #print (len(train_label_batch))
            #print (len(train_label_batch[0]))
            #print (len(train_label_batch[0][0]))
            #sys.exit()
            
            for i in range(int(len(train_image)/batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = batch_size*i
                sess.run(optimizer, feed_dict={
                    images_placeholder: train_image_batch[batch:batch+batch_size],
                    labels_placeholder: train_label_batch[batch:batch+batch_size],
                    keep_prob1: 1.0,
                    keep_prob2: 1.0,
                    keep_prob3: 1.0,
                    keep_prob4: 1.0,
                    keep_prob5: 1.0,
                    keep_prob6: 0.5,
                    keep_prob7: 1.0
                })
            
            # 1 step終わるたびに訓練データのロスを計算する
            train_loss = 0
            for i in range(int(len(train_image)/CALC_BATCH)):
                batch = CALC_BATCH*i
                train_loss = train_loss + sess.run(cost, feed_dict={
                    images_placeholder: train_image[batch:batch+CALC_BATCH],
                    labels_placeholder: train_label[batch:batch+CALC_BATCH],
                    keep_prob1: 1.0,
                    keep_prob2: 1.0,
                    keep_prob3: 1.0,
                    keep_prob4: 1.0,
                    keep_prob5: 1.0,
                    keep_prob6: 1.0,
                    keep_prob7: 1.0
                })
            train_loss = train_loss/(len(train_image)/CALC_BATCH)
            print ("train loss %g"%( train_loss))
            fp_train.write("%d %g\n" %(step,train_loss))
        
            # 1 step終わるたびにテストデータのロスを計算する
            test_loss = 0
            for i in range(int(len(test_image)/CALC_BATCH)):
                batch = CALC_BATCH*i
                test_loss = test_loss + sess.run(cost, feed_dict={
                    images_placeholder: test_image[batch:batch+CALC_BATCH],                
                    labels_placeholder: test_label[batch:batch+CALC_BATCH],
                    keep_prob1: 1.0,
                    keep_prob2: 1.0,
                    keep_prob3: 1.0,
                    keep_prob4: 1.0,
                    keep_prob5: 1.0,
                    keep_prob6: 1.0,
                    keep_prob7: 1.0
                })
            test_loss = test_loss/(len(test_image)/CALC_BATCH)
            print ("test loss %g"%( test_loss))
            fp_test.write("%d %g\n" %(step,test_loss))
        
        
            #ユークリッド距離
            # 1 step終わるたびに訓練データのユークリッド距離を計算する
            for i in range(int(len(train_image)/CALC_BATCH)):
                batch = CALC_BATCH*i
                if i==0:
                    train_estimation = sess.run(pred, feed_dict={
                        images_placeholder: train_image[batch:batch+CALC_BATCH],
                        keep_prob1: 1.0,
                        keep_prob2: 1.0,
                        keep_prob3: 1.0,
                        keep_prob4: 1.0,
                        keep_prob5: 1.0,
                        keep_prob6: 1.0,
                        keep_prob7: 1.0
                    })
                else:
                    train_estimation = np.vstack((train_estimation,sess.run(pred, feed_dict={
                        images_placeholder: train_image[batch:batch+CALC_BATCH],
                        keep_prob1: 1.0,
                        keep_prob2: 1.0,
                        keep_prob3: 1.0,
                        keep_prob4: 1.0,
                        keep_prob5: 1.0,
                        keep_prob6: 1.0,
                        keep_prob7: 1.0
                    })))
            
            #print (len(train_estimation))#1000
            #print (len(train_estimation[0]))#10
            #print (len(train_estimation[0][0]))#2
            #print (train_estimation[0])
            #print (train_label[0])
            #sys.exit()
        
            train_loss_table = np.zeros([len(train_estimation),n_steps])
        
            for i in range(len(train_estimation)):
                for j in range(n_steps):        
                    euclidean_dis = np.sqrt(np.power(train_estimation[i][j]-train_label[i][j], 2).sum())
                    train_loss_table[i][j] = euclidean_dis

            train_loss_seq_ave = train_loss_table.mean(axis = 0)
            train_loss_ave = train_loss_seq_ave.mean()
        
            #print ("train_euclidian_loss_seq = ",train_loss_seq_ave)
            fp_train_sqrt_all.write("%d " %(step))
            for i in range(n_steps):
                fp_train_sqrt_all.write("%g" %(train_loss_seq_ave[i]))
                if not (i==n_steps-1):
                    fp_train_sqrt_all.write(" ")
            fp_train_sqrt_all.write("\n")
        
            print ("train_euclidian_loss_ave ",train_loss_ave)
            fp_train_sqrt_ave.write("%d %g\n" %(step,train_loss_ave))

            #sys.exit()
        
            
            # 1 step終わるたびにテストデータのユークリッド距離を計算する
            for i in range(int(len(test_image)/CALC_BATCH)):
                batch = CALC_BATCH*i
                if i==0:
                    test_estimation = sess.run(pred, feed_dict={
                        images_placeholder: test_image[batch:batch+CALC_BATCH],
                        keep_prob1: 1.0,
                        keep_prob2: 1.0,
                        keep_prob3: 1.0,
                        keep_prob4: 1.0,
                        keep_prob5: 1.0,
                        keep_prob6: 1.0,
                        keep_prob7: 1.0
                    })
                else:
                    test_estimation = np.vstack((test_estimation,sess.run(pred, feed_dict={
                        images_placeholder: test_image[batch:batch+CALC_BATCH],
                        keep_prob1: 1.0,
                        keep_prob2: 1.0,
                        keep_prob3: 1.0,
                        keep_prob4: 1.0,
                        keep_prob5: 1.0,
                        keep_prob6: 1.0,
                        keep_prob7: 1.0
                    })))
        
            #print (len(test_estimation))#200
            #print (len(test_estimation[0]))#10
            #print (len(test_estimation[0][0]))#2
            #print (test_estimation[0])
            #print (test_label[0])
            #sys.exit()
        
            test_loss_table = np.zeros([len(test_estimation),n_steps])
        
            for i in range(len(test_estimation)):
                for j in range(n_steps):        
                    euclidean_dis = np.sqrt(np.power(test_estimation[i][j]-test_label[i][j], 2).sum())
                    test_loss_table[i][j] = euclidean_dis
        
            test_loss_seq_ave = test_loss_table.mean(axis = 0)
            test_loss_ave = test_loss_seq_ave.mean()
        
            #print ("test_euclidian_loss_seq = ",loss_seq_ave)
            fp_test_sqrt_all.write("%d " %(step))
            for i in range(n_steps):
                fp_test_sqrt_all.write("%g" %(test_loss_seq_ave[i]))
                if not (i==n_steps-1):
                    fp_test_sqrt_all.write(" ")
            fp_test_sqrt_all.write("\n")
        
            print ("test_euclidian_loss_ave ",test_loss_ave)
            fp_test_sqrt_ave.write("%d %g\n" %(step,test_loss_ave))
        
            
            #SAVE_MODEL_STEPステップごとにモデルを保存
            if step!=0 and (step+1)%SAVE_MODEL_STEP==0:
                MODEL_PATH = "./model/rnn_model"+str(exp)+"-"+str(num)+"-"+str(step)+".ckpt"
                save_path = saver.save(sess, MODEL_PATH)

            
            
        
        print("Optimization Finished!")
    
        #出力結果
        #train
        #print (len(train_estimation))#1000
        #print (len(train_estimation[0]))#10
        #print (len(train_estimation[0][0]))#2
        
        #train
        f = open(result_train_dir,"w")
        for i in range(TRAIN_SIZE):
            for j in range(n_steps):
                for k in range(n_classes):
                    f.write("%g" %(train_estimation[i][j][k]))
                    f.write(" ")
                for k in range(n_classes):
                    f.write("%g" %(train_label[i][j][k]))
                    if not (k==n_classes-1):
                        f.write(" ")
                f.write("\n")
        f.close()
    
        #test
        f = open(result_test_dir,"w")
        for i in range(TEST_SIZE):
            for j in range(n_steps):
                for k in range(n_classes):
                    f.write("%g" %(test_estimation[i][j][k]))
                    f.write(" ")
                for k in range(n_classes):
                    f.write("%g" %(test_label[i][j][k]))
                    if not (k==n_classes-1):
                        f.write(" ")
                f.write("\n")
        f.close()
        
        
        # 最終的なモデルを保存
        MODEL_PATH = "./model/rnn_model"+str(exp)+"-"+str(num)+"-"+str(step)+".ckpt"
        save_path = saver.save(sess, MODEL_PATH)

        # file close
        fp_train.close()
        fp_test.close()    
        fp_train_sqrt_all.close()    
        fp_train_sqrt_ave.close()
        fp_test_sqrt_all.close()    
        fp_test_sqrt_ave.close()
        
        num = num+1



