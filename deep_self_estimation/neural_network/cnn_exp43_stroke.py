#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import tensorflow.compat.v1 as tf

np.random.seed(2323)

exp = 43

MAXSTEP = 1000

BATCHSIZE = 20

CALC_BATCH = 1000

SAVE_MODEL_STEP = 100

TRAIN_SIZE = 10000
TEST_SIZE = 2000

NUM_MAX = int((TRAIN_SIZE+TEST_SIZE)/TEST_SIZE)
NUM_START = 0

NUM_CLASSES = 2 #座標x,y

IMAGE_ROW = 48
IMAGE_COL = 48
IMAGE_PIXELS = IMAGE_ROW*IMAGE_COL*3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '../screenshot/scsho_txt/rnn'+str(43)+'.txt', 'File name of data dir')

flags.DEFINE_integer('max_steps', MAXSTEP, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', BATCHSIZE, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
#flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')


def inference(images_placeholder, keep_prob1, keep_prob2, keep_prob3, keep_prob4, keep_prob5, keep_prob6, keep_prob7 ):

    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      #VALIDはみだしなし

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    # 入力を28x28x3に変形
    x_image = tf.reshape(images_placeholder, [-1,IMAGE_ROW ,IMAGE_COL , 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        #ドロップアウト1　畳み込み1のインプット
        x_image_drop = tf.nn.dropout(x_image, keep_prob1)
        h_conv1 = tf.nn.relu(conv2d(x_image_drop, W_conv1) + b_conv1)
        
        
    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        #ドロップアウト2　プーリング層1のインプット
        h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob2)
        h_pool1 = max_pool_2x2(h_conv1_drop)
    
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        #ドロップアウト3　畳み込み層2のインプット
        h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob3)
        h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        #ドロップアウト4 プーリング層2のインプット
        h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob4)
        h_pool2 = max_pool_2x2(h_conv2_drop)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        #W_fc1 = weight_variable([7*7*64, 1024])
        W_fc1 = weight_variable([int(IMAGE_ROW/4*IMAGE_COL/4*64), 1024])
        b_fc1 = bias_variable([1024])
        
        #h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_pool2_flat = tf.reshape(h_pool2, [-1,int(IMAGE_ROW/4*IMAGE_COL/4*64)])
        #ドロップアウト5 全結合層1のインプット
        h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat,keep_prob5)
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat_drop, W_fc1) + b_fc1)
        
    # 全結合層2の作成
    with tf.name_scope('output') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        
        #ドロップアウト6 全結合層２のインプット
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob6)
        
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        #ドロップアウト7 全結合層2のアウトプット
        y_conv_drop = tf.nn.dropout(y_conv, keep_prob7)

    return y_conv_drop



def loss(logits, labels):
    # 二乗誤差の平方根の計算
    squared_error = tf.reduce_mean(tf.square(labels-logits))

    return squared_error


def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step



if __name__ == '__main__':
    # ファイルを開く
    f = open(FLAGS.dataset, 'r')
    # データを入れる配列
    dataset_image = []
    dataset_label = []
    
    print("...loading data")
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # 画像読み込み
        img = cv2.imread(l[0])
        #正方形にトリミング
        img = img[:,400-240:400+240]
        img = cv2.resize(img, (IMAGE_ROW, IMAGE_COL))
        # 一列にした後、0-1のfloat値にする
        dataset_image.append(img.flatten().astype(np.float32)/255.0)
        x = float(l[1])
        y = float(l[3])
        z = float(l[2])
        angle = float(l[4])
        pitch = float(l[5])

        #クラス2:位置のみ推定
        tmp = np.zeros(NUM_CLASSES)
        tmp[0] = x
        tmp[1] = y
        dataset_label.append(tmp)
    
    
    f.close()
    
    DATA_SIZE = len(dataset_label)
    dataset_image = np.asarray(dataset_image)
    dataset_label = np.asarray(dataset_label)       
    print(len(dataset_image))
    print(len(dataset_label))

    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        print("...building graph")
        # 画像を入れる仮のTensor
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))

        # dropout率を入れる仮のTensor
        keep_prob1 = tf.placeholder("float")
        keep_prob2 = tf.placeholder("float")        
        keep_prob3 = tf.placeholder("float")        
        keep_prob4 = tf.placeholder("float")        
        keep_prob5 = tf.placeholder("float")        
        keep_prob6 = tf.placeholder("float")        
        keep_prob7 = tf.placeholder("float")
        
        # inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob1, keep_prob2, keep_prob3, keep_prob4, keep_prob5, keep_prob6, keep_prob7)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        
        # training()を呼び出して訓練
        train_op = training(loss_value, FLAGS.learning_rate)
        
        # 保存の準備
        saver = tf.train.Saver(max_to_keep = 0)
        
        # Sessionの作成
        sess = tf.Session(config=config)
        # 変数の初期化
        sess.run(tf.initialize_all_variables())

        
        #roop
        num = NUM_START
        while num<NUM_MAX:
            
            MODEL_PATH = "./cnn_model/cnn_model"+str(exp)+"-"+str(num)+".ckpt" 
            
            traindir = "./cnn_data_exp/exp"+str(exp)+"-train-"+str(num)+".txt"
            testdir = "./cnn_data_exp/exp"+str(exp)+"-test-"+str(num)+".txt"
            traindir_sqrt = "./cnn_data_exp/exp"+str(exp)+"-train-sqrt-"+str(num)+".txt"
            testdir_sqrt = "./cnn_data_exp/exp"+str(exp)+"-test-sqrt-"+str(num)+".txt"            
            result_train_dir = "./cnn_data_result/train_exp"+str(exp)+"-"+str(num)+".txt"
            result_test_dir = "./cnn_data_result/test_exp"+str(exp)+"-"+str(num)+".txt"
                     
            # 変数の初期化
            sess.run(tf.initialize_all_variables())
            
            
            #交差検証法            
            k = int(DATA_SIZE/TEST_SIZE)
            train_image = np.asarray(np.vstack((dataset_image[:TEST_SIZE*num,],
                                                dataset_image[TEST_SIZE*(num+1):,])))
            train_label = np.asarray(np.vstack((dataset_label[:TEST_SIZE*num,],
                                                dataset_label[TEST_SIZE*(num+1):,])))

            test_image = np.asarray(dataset_image[TEST_SIZE*num:TEST_SIZE*(num+1),])
            test_label = np.asarray(dataset_label[TEST_SIZE*num:TEST_SIZE*(num+1),])

            print (len(train_image))
            print (len(test_image))

            
            fp1 = open(traindir,"w")
            fp2 = open(testdir,"w")
            fp3 = open(traindir_sqrt,"w")
            fp4 = open(testdir_sqrt,"w")
            
            
            # 訓練の実行
            print("...start training")
            for step in range(int(FLAGS.max_steps)):
                print ("")
                print (num,"/",k,("step:{0}".format(step)))

                #シャッフル系列作成
                random_seq = np.random.permutation(len(train_image))
                train_image_batch = []
                train_label_batch = []
                for i in range(int(len(train_image)/FLAGS.batch_size)):
                    batch = FLAGS.batch_size*i
                    for j in range(FLAGS.batch_size):
                        train_image_batch.append(train_image[random_seq[batch+j]])
                        train_label_batch.append(train_label[random_seq[batch+j]])
                                             
                train_image_batch = np.asarray(train_image_batch)
                train_label_batch = np.asarray(train_label_batch)
                
                for i in range(int(len(train_image)/FLAGS.batch_size)):
                    # batch_size分の画像に対して訓練の実行
                    batch = FLAGS.batch_size*i
                    # feed_dictでplaceholderに入れるデータを指定する
                    sess.run(train_op, feed_dict={
                        images_placeholder: train_image_batch[batch:batch+FLAGS.batch_size],
                        labels_placeholder: train_label_batch[batch:batch+FLAGS.batch_size],
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
                    train_loss = train_loss + sess.run(loss_value, feed_dict={
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
                #print "step %d, train accuracy %g"%(step, train_accuracy)
                train_loss = train_loss/(len(train_image)/CALC_BATCH)
                print ("train loss %g"%( train_loss))
                fp1.write("%d %g\n" %(step,train_loss))
                
               
                # 1 step終わるたびにテストデータのロスを計算する
                test_loss = 0
                for i in range(int(len(test_image)/CALC_BATCH)):
                    batch = CALC_BATCH*i
                    test_loss = test_loss + sess.run(loss_value, feed_dict={
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
                fp2.write("%d %g\n" %(step,test_loss))
                
                
                #ユークリッド距離
                # 1 step終わるたびに訓練データのユークリッド距離を計算する    
                for i in range(int(len(train_image)/CALC_BATCH)):
                    batch = CALC_BATCH*i
                    if i==0:
                        train_estimation = sess.run(logits, feed_dict={
                            images_placeholder: train_image[batch:batch+CALC_BATCH],
                            keep_prob1: 1.0,
                            keep_prob2: 1.0,                        
                            keep_prob3: 1.0,
                            keep_prob4: 1.0,                        
                            keep_prob5: 1.0,                        
                            keep_prob6: 1.0,                        
                            keep_prob7: 1.0
                        })
                    
                    #print train_estimation
                    else:
                        train_estimation = np.vstack((train_estimation,sess.run(logits, feed_dict={
                            images_placeholder: train_image[batch:batch+CALC_BATCH],
                            keep_prob1: 1.0,
                            keep_prob2: 1.0,                        
                            keep_prob3: 1.0,
                            keep_prob4: 1.0,                        
                            keep_prob5: 1.0,                        
                            keep_prob6: 1.0,                        
                            keep_prob7: 1.0
                        })))
                    #sys.exit()
                
                train_loss_table = np.zeros([len(train_estimation)])
                for i in range(len(train_estimation)):
                    euclidean_dis = np.sqrt(np.power(train_estimation[i]-train_label[i], 2).sum())
                    train_loss_table[i] = euclidean_dis

                train_loss_sqrt_ave = train_loss_table.mean()
                print ("train loss sqrt ",train_loss_sqrt_ave)
                fp3.write("%d %g\n" %(step,train_loss_sqrt_ave))
                
        
                # 1 step終わるたびにテストデータのユークリッド距離を計算する
                for i in range(int(len(test_image)/CALC_BATCH)):
                    batch = CALC_BATCH*i
                    if i==0:
                        test_estimation = sess.run(logits, feed_dict={
                            images_placeholder: test_image[batch:batch+CALC_BATCH],
                            keep_prob1: 1.0,
                            keep_prob2: 1.0,                        
                            keep_prob3: 1.0,
                            keep_prob4: 1.0,                        
                            keep_prob5: 1.0,                        
                            keep_prob6: 1.0,                        
                            keep_prob7: 1.0
                        })
                    
                    #print train_estimation
                    else:
                        test_estimation = np.vstack((test_estimation,sess.run(logits, feed_dict={
                            images_placeholder: test_image[batch:batch+CALC_BATCH],
                            keep_prob1: 1.0,
                            keep_prob2: 1.0,                        
                            keep_prob3: 1.0,
                            keep_prob4: 1.0,                        
                            keep_prob5: 1.0,                        
                            keep_prob6: 1.0,                        
                            keep_prob7: 1.0
                        })))

                test_loss_table = np.zeros([len(test_estimation)])
                for i in range(len(test_estimation)):
                    euclidean_dis = np.sqrt(np.power(test_estimation[i]-test_label[i], 2).sum())
                    test_loss_table[i] = euclidean_dis
                
                test_loss_sqrt_ave = test_loss_table.mean()
                print ("test loss sqrt ",test_loss_sqrt_ave)
                fp4.write("%d %g\n" %(step,test_loss_sqrt_ave))

                
                #SAVE_MODEL_STEPステップごとにモデルを保存
                if step!=0 and (step+1)%SAVE_MODEL_STEP==0:
                    MODEL_PATH = "./cnn_model/cnn_model"+str(exp)+"-"+str(num)+"-"+str(step)+".ckpt" 
                    save_path = saver.save(sess, MODEL_PATH)
                    
            
            #出力結果
            f = open(result_train_dir,"w")
            for i in range(TRAIN_SIZE):
                for j in range(NUM_CLASSES):
                    f.write(str(train_estimation[i][j]))
                    f.write(" ")
                for j in range(NUM_CLASSES):
                    f.write(str(train_label[i][j]))
                    if not (j==NUM_CLASSES-1):
                        f.write(" ")
                f.write("\n")        
            f.close()
            
            
            f = open(result_test_dir,"w")
            for i in range(TEST_SIZE):
                for j in range(NUM_CLASSES):
                    f.write(str(test_estimation[i][j]))
                    f.write(" ")
                for j in range(NUM_CLASSES):
                    f.write(str(test_label[i][j]))
                    if not (j==NUM_CLASSES-1):
                        f.write(" ")
                f.write("\n")
            f.close()
            
            # 最終的なモデルを保存
            MODEL_PATH = "./cnn_model/cnn_model"+str(exp)+"-"+str(num)+"-"+str(step)+".ckpt" 
            save_path = saver.save(sess, MODEL_PATH)
            fp1.close()
            fp2.close()
            fp3.close()
            fp4.close()            
            
            num = num+1
    
