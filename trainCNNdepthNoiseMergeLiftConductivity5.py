# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

batch_size = 100
training_epochs = 10000
display_epochs = 100
dataset_size = 20000

lossDict = {}

for coil in [1, 3, 5, 7]:
    coilLoss = {}
    for frequency in [25, 100, 400]:
        fname = "dataset_plus_fre" + str(frequency)
        dataset = pickle.load(open("data/" + fname + ".pickle", "rb"))
        dataset = dataset[coil]
        datasetDummy = []
        for lift in [1, 3, 5]:
            datasetDummy.extend(dataset[lift])
        dataset = datasetDummy

        # 導電率5未満のみ使用
        datasetDummy = []
        for data in dataset:
            if data[5] < 5:
                datasetDummy.append(data)
        dataset = datasetDummy
        datasetMatrix = np.zeros((dataset_size, dataset[0].shape[0]))
        for i in range(dataset_size):
            data = dataset[random.randint(0, len(dataset)-1)]
            for k in range(99):
                if k < 6:
                    datasetMatrix[i, k] = data[k]
                else:
                    datasetMatrix[i, k] = data[k] * (random.random() * 0.2 + 0.9)
        dataset = datasetMatrix

        widthRMS = 0.5
        depthRMS = 10
        coilRMS = 7
        liftRMS = 5
        frequencyRMS = 400
        conductivityRMS = 100
        # 規格化
        dataset[:, 0] /= widthRMS
        dataset[:, 1] /= depthRMS
        dataset[:, 2] /= coilRMS
        dataset[:, 3] /= liftRMS
        dataset[:, 4] /= frequencyRMS
        dataset[:, 5] /= conductivityRMS
        dataset[:, 6:] -= np.min(dataset[:, 6:])
        dataset[:, 6:] /= np.max(dataset[:, 6:])
        # 型変換
        dataset = np.array(dataset, dtype=np.float32)
        # datasetを分ける
        datasetTrain = dataset[:dataset_size-1000, :]
        datasetTest = dataset[dataset_size-1000:, :]
        pickle.dump(datasetTrain, open("data/" + fname + "_" + str(coil) + "_0_Train_NoiseMergeLiftConductivity5.pickle", "wb"))
        pickle.dump(datasetTest, open("data/" + fname + "_" + str(coil) + "_0_Test_NoiseMergeLiftConductivity5.pickle", "wb"))

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

        sess.run(tf.initialize_all_variables())

        for i in range(training_epochs):
            for k in range(0, len(datasetTrain), batch_size):
                batch = np.zeros((batch_size, 93))
                batch = np.array(batch, dtype=np.float32)
                batch[:, :] = datasetTrain[k:k+batch_size, 6:]
                output = datasetTrain[k:k+batch_size, 1:2]
                train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
            
            if i%display_epochs == 0:
                batch = np.zeros((datasetTrain.shape[0], 93))
                batch = np.array(batch, dtype=np.float32)
                batch[:, :] = datasetTrain[:, 6:]
                output = datasetTrain[:, 1:2]
                train_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
                batch = np.zeros((datasetTest.shape[0], 93))
                batch = np.array(batch, dtype=np.float32)
                batch[:, :] = datasetTest[:, 6:]
                output = datasetTest[:, 1:2]
                test_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
                print(str(i) + "epochs_finished!")
                print("train_loss===" + str(train_loss))
                print("test_loss===" + str(test_loss))
        saver.save(sess, "./model2/20181121CNNmodeldepthNoiseMergeLift" + fname + "_" + str(coil) + "_by5")
        sess.close()
        coilLoss[frequency] = (train_loss, test_loss)
    lossDict[coil] = coilLoss
# pickle.dump(lossDict, open("lossDict20181114CNNmodel_dataset_plus_con0_noise_merge_lift.pickle", "wb"))