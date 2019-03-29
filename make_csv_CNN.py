# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import csv

header = ['label', 'prediction']

for coil in [1, 3, 5, 7]:
        for frequency in [25, 100, 400]:
            
            fname = "dataset_plus_fre" + str(frequency)
            
            dataset = pickle.load(open("data/" + fname + "_" + str(coil) + "_0_Test_NoiseMergeLiftConductivity5.pickle", "rb"))
            
            tf.reset_default_graph()
    
            x = tf.placeholder("float", shape=[None, 93])
            y_ = tf.placeholder("float", shape=[None, 1])
    
            # 荷重作成
            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)
    
            # バイアス作成
            def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)
    
            # 畳み込み処理を定義
            def conv2d_pad(x, W):
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
            # プーリング処理を定義
            def max_pool_2_2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
    
            # 畳み込み層1
            W_conv1 = weight_variable([3, 3, 1, 256])
            b_conv1 = bias_variable([256])
            x_image = tf.reshape(x, [-1, 31, 3, 1])
            h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
            # プーリング層1
            h_pool1 = max_pool_2_2(h_conv1)
    
            # 畳み込み層2
            W_conv2 = weight_variable([3, 2, 256, 256])
            b_conv2 = bias_variable([256])
            h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
            # プーリング層2
            h_pool2 = max_pool_2_2(h_conv2)
    
            # 全結合層1
            W_fc1 = weight_variable([256*8, 1024])
            b_fc1 = bias_variable([1024])
            h_flat = tf.reshape(h_pool2, [-1, 256*8])
            h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
    
            # ドロップアウト
            keep_prob = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
            # 全結合層2
            W_fc2 = weight_variable([1024, 1])
            b_fc2 = bias_variable([1])
            y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
            # 学習誤差を求める
            loss = tf.reduce_mean(tf.square(y_ - y_out))
    
            # 最適化処理
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
            sess = tf.Session()
    
            saver = tf.train.Saver()
            saver.restore(sess, "./model2/20181121CNNmodeldepthNoiseMergeLift" + fname + "_" + str(coil) + "_by5")
                
            batch = np.zeros((dataset.shape[0], 93))
            batch = np.array(batch, dtype=np.float32)
            batch[:, :] = dataset[:, 6:]
            output = dataset[:, 1:2]
            prediction = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
            
            # ソート
            indexes = np.argsort(dataset[:, 1])
            plotList = []
            for i in range(len(indexes)):
                index = indexes[i]
                if i == 0:
                    plots = []
                elif dataset[index, 1] != dataset[indexes[i-1], 1]:
                    plotList.append(plots)
                    plots = []
                plots.append(prediction[index, 0])
            plotList.append(plots)
            # pickle.dump(plotList, open("plotListCNN" + fname + "_" + str(coil) + ".pickle", "wb"))
            
            body = []
            for i in range(len(plotList)):
                plots = plotList[i]
                for plot in plots:
                    if i == 0:
                        body.append([0.2, plot*10.0])
                    elif i == 1:
                        body.append([0.4, plot*10.0])
                    elif i == 2:
                        body.append([0.6, plot*10.0])
                    elif i == 3:
                        body.append([0.8, plot*10.0])
                    elif i == 4:
                        body.append([1.0, plot*10.0])
                    elif i == 5:
                        body.append([1.5, plot*10.0])
                    elif i == 6:
                        body.append([2.0, plot*10.0])
                    elif i == 7:
                        body.append([2.5, plot*10.0])
                    elif i == 8:
                        body.append([3.0, plot*10.0])
                    elif i == 9:
                        body.append([4.0, plot*10.0])
                    elif i == 10:
                        body.append([5.0, plot*10.0])
                    elif i == 11:
                        body.append([7.0, plot*10.0])
                    elif i == 12:
                        body.append([10.0, plot*10.0])
            with open("plot_cnn_coil" + str(coil) + "_liftmerge_frequency" + str(frequency) + "_by5.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(body)        