# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

fname = "dataset_plus_fre25"
dataset25 = pickle.load(open("data/" + fname + ".pickle", "rb"))
fname = "dataset_plus_fre100"
dataset100 = pickle.load(open("data/" + fname + ".pickle", "rb"))
fname = "dataset_plus_fre400"
dataset400 = pickle.load(open("data/" + fname + ".pickle", "rb"))

dataset25_1_1_0 = dataset25[1][1]
dataset25_1_3_0 = dataset25[1][3]
dataset25_1_5_0 = dataset25[1][5]
dataset100_1_1_0 = dataset100[1][1]
dataset100_1_3_0 = dataset100[1][3]
dataset100_1_5_0 = dataset100[1][5]
dataset400_1_1_0 = dataset400[1][1]
dataset400_1_3_0 = dataset400[1][3]
dataset400_1_5_0 = dataset400[1][5]

datasetDummy = []
for data in dataset25_1_1_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset25_1_1_0 = datasetDummy
datasetDummy = []
for data in dataset25_1_3_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset25_1_3_0 = datasetDummy
datasetDummy = []
for data in dataset25_1_5_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset25_1_5_0 = datasetDummy
datasetDummy = []
for data in dataset100_1_1_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset100_1_1_0 = datasetDummy
datasetDummy = []
for data in dataset100_1_5_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset100_1_5_0 = datasetDummy
datasetDummy = []
for data in dataset100_1_3_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset100_1_3_0 = datasetDummy
datasetDummy = []
for data in dataset400_1_1_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset400_1_1_0 = datasetDummy
datasetDummy = []
for data in dataset400_1_3_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset400_1_3_0 = datasetDummy
datasetDummy = []
for data in dataset400_1_5_0:
    if data[5] == 0:
        datasetDummy.append(data)
dataset400_1_5_0 = datasetDummy

img_merge25_1_1_0 = np.zeros((31, 27))
for i in range(9):
    img_merge25_1_1_0[:, 3*i:3*i+3] = np.reshape(dataset25_1_1_0[i][6:], [31, 3])
img_merge25_1_3_0 = np.zeros((31, 27))
for i in range(9):
    img_merge25_1_3_0[:, 3*i:3*i+3] = np.reshape(dataset25_1_3_0[i][6:], [31, 3])
img_merge25_1_5_0 = np.zeros((31, 27))
for i in range(9):
    img_merge25_1_5_0[:, 3*i:3*i+3] = np.reshape(dataset25_1_5_0[i][6:], [31, 3])
img_merge100_1_1_0 = np.zeros((31, 27))
for i in range(9):
    img_merge100_1_1_0[:, 3*i:3*i+3] = np.reshape(dataset100_1_1_0[i][6:], [31, 3])
img_merge100_1_3_0 = np.zeros((31, 27))
for i in range(9):
    img_merge100_1_3_0[:, 3*i:3*i+3] = np.reshape(dataset100_1_3_0[i][6:], [31, 3])
img_merge100_1_5_0 = np.zeros((31, 27))
for i in range(9):
    img_merge100_1_5_0[:, 3*i:3*i+3] = np.reshape(dataset100_1_5_0[i][6:], [31, 3])
img_merge400_1_1_0 = np.zeros((31, 27))
for i in range(9):
    img_merge400_1_1_0[:, 3*i:3*i+3] = np.reshape(dataset400_1_1_0[i][6:], [31, 3])
img_merge400_1_3_0 = np.zeros((31, 27))
for i in range(9):
    img_merge400_1_3_0[:, 3*i:3*i+3] = np.reshape(dataset400_1_3_0[i][6:], [31, 3])
img_merge400_1_5_0 = np.zeros((31, 27))
for i in range(9):
    img_merge400_1_5_0[:, 3*i:3*i+3] = np.reshape(dataset400_1_5_0[i][6:], [31, 3])