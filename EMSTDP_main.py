import os
from cv2 import log

from scipy.ndimage.measurements import labeled_comprehension
from scipy.special import softmax

import sys
# import pyximport; pyximport.install()

import numpy as np
import EMSTDP_algo as svp
# import BPSNN_allSpikeErr_batcheventstdp2 as svp
import matplotlib.pyplot as plt
import pickle
import gzip
import time
import random
import numpy as np
import sklearn.metrics as sk
import cv2

from tqdm import tqdm, tqdm_gui, tnrange, tgrange, trange
# import progressbar
import torchvision.transforms as trn
import torch
from utils import get_measures
from utils import print_measures
path = os.getcwd()

train_data = np.load(path +"/em_data/em_t.npy")
train_label = np.load(path +"/em_data/em_l.npy")

test_data = np.load(path +"/em_data/em_test.npy")
test_label = np.load(path +"/em_data/em_lest.npy")

ood_data = np.load(path +"/em_data/em_tood.npy")
ood_label = np.load(path +"/em_data/em_lood.npy")

dense_size = 32*32
dataTest = test_data
labelTest = test_label
dataTest = np.reshape(dataTest, (1,len(dataTest),dense_size))



ood_data = np.reshape(ood_data[:548], (1,548,dense_size))
ood_label = ood_label[:548]

# '''
# change here
# '''
# #labelTest = np.argmax(labelTest, axis = 1)
# #print(labelTest.shape)

total_train_size = len(train_data)
data = np.reshape(train_data, (1,len(train_data),dense_size))
label = train_label
data_index = (np.linspace(0, total_train_size - 1, total_train_size)).astype(int)



# initialize hyper-parameters (the descriptions are in the Network class)
h = [200]  # [100,300,500,700,test_size,1500]

ind = -1
epochs = 100
T = 100
twin = int( T / 2 - 1)
epsilon = 3
scale = 1.0
bias = 0.0
batch_size = 10
tbs = 100
fr = 1.0
rel = 0
delt = 5
clp = True
lim = 1.0
dropr = 0.0

final_energy = np.zeros([epochs])

hiddenThr1 = 0.5
outputThr1 = 0.1
dense_size = 1024
train_size = len(train_data)
test_size = len(test_data)
energies = np.zeros([train_size])
batch_energy = np.zeros([int(train_size / 50)])  # bach_size = 50
ind += 1
acc = []

tmp_rand = np.random.random([T, 1, 1])
randy = np.tile(tmp_rand, (1, batch_size, dense_size))
tmp_d = np.zeros([T, batch_size, dense_size])

lr = 0.003
online_norm = False
# def __init__(self, dfa, dropr, evt, norm, rel, delt, dr, init, clp, lim, inputs, hiddens, outputs, threshold_h, threshold_o, T=100, bias=0.0, lr=0.0001, scale=1.0, twin=100, epsilon=2):
snn_network = svp.Network(0, dropr, 0, 0.0, rel, delt, 1, 0, clp, lim, dense_size, h, 2, hiddenThr1*fr, outputThr1*fr, T, bias, lr, scale, twin, epsilon,

                         online_norm= False, train_oe= True)

train_acc =[]
test_acc = []
## load trained weights, load_weight=True
load_weight = False
load_epoch = 1

ver_period = 1000
s_index = data_index
# for ep in trange(epochs):
# for ep in range(epochs):
#     snn_network.lr = 1.0 / (1.0 * (5000.0+ ep ))
#     pred1 = np.zeros([train_size])
#     # np.random.shuffle(s_index)
#     spikes = np.zeros([T, batch_size, dense_size]).astype(float)
#     spikes2 = np.zeros([T, batch_size, dense_size]).astype(float)
#     # for i in trange(train_size / batch_size, leave=False):
#     for i in trange(int (train_size / batch_size)):
#         if ((i + 1) * batch_size % ver_period == 0):  # 5000
#             pred = np.zeros([int(test_size)])
#             for i2 in range(int(test_size / tbs)):  # train_size

#                 tmp_rand = np.random.random([T, 1, 1])
#                 randy = np.tile(tmp_rand, (1, tbs, dense_size))

#                 tmp_d = np.tile(dataTest[:, i2 * tbs:(i2 + 1) * tbs, :], (T, 1, 1))
#                 spikes2 = randy < (tmp_d * fr)

#                 _,pred[i2 * tbs:(i2 + 1) * tbs] = snn_network.Test(spikes2.astype(float), tbs)
#             acn = sum(pred == labelTest[:test_size]) / float(test_size)
#             print( str(ep) + " test_accuray " + str(acn) + " LR " + str(snn_network.lr))
#             test_acc.append(acn)
#             acc.append(sum(pred == labelTest[:test_size]) / float(test_size))

#         tmp_rand = np.random.random([T, 1, 1])
#         randy = np.tile(tmp_rand, (1, batch_size, dense_size))
#         tmp_d = np.tile(data[:, s_index[i * batch_size:(i + 1) * batch_size], :], (T, 1, 1))
#         spikes = randy < (tmp_d * fr)
#         pred1[i * batch_size:(i + 1) * batch_size], energies[i] = snn_network.Train(spikes.astype(float), (
#         label[s_index[i * batch_size:(i + 1) * batch_size]]), batch_size)
#     acn = sum(pred1 == label[s_index[:train_size]]) / float(train_size)
#     train_acc.append(acn)

#     print(str(ep) + " train_accuray " + str(acn))
#     np.save("w_h.npy", snn_network.w_h)
#     np.save("w_o.npy",snn_network.w_o)
# np.save("train_acc.npy", train_acc)
# np.save("test_acc.npy", test_acc)

w_h = np.load("w_h.npy")
w_o = np.load("w_o.npy")
snn_network.w_h = w_h
snn_network.w_o = w_o
# oe snn, first to train outlier exposure number, the outlier exposure number should be 1/5 
# of the indistribution sample
# epoch 50, batch size is 1

from keras.datasets import mnist 
all_data = mnist.load_data()
TRAINING = all_data[0]
TESTING = all_data[1]

_, h, w = np.shape(TRAINING[0])

train_size = 100                          # verification period
test_size = 100                          # verification test size
ver_period = train_size

epochs = 50                              # number of epochs
data_oe = TRAINING[0][0:100]
data_oe = np.reshape(data_oe, (len(data_oe),28,28))
data_ood = np.zeros((len(data_oe),32,32))

for i in range(len(data_oe)):
    img = data_oe[i]
    res = cv2.resize(img,dsize=(32,32))
    data_ood[i] = res

data_ood = np.reshape(data_ood, (1,len(data_ood),dense_size)).astype(float)/255.0



def train_oe(epochs = 50, batch_size =1, data = data_ood, dense_size = 32*32):
    for ep in range(epochs):
        snn_network.lr = 1.0 / (1.0 * (5000.0+ ep ))
        train_size = data.shape[1]
        pred1 = np.zeros([train_size])
        # np.random.shuffle(s_index)
        spikes = np.zeros([T, batch_size, dense_size]).astype(float)
        for i in trange(int (train_size / batch_size)):
            tmp_rand = np.random.random([T, 1, 1])
            randy = np.tile(tmp_rand, (1, batch_size, dense_size))
            tmp_d = np.tile(data[:, s_index[i * batch_size:(i + 1) * batch_size], :], (T, 1, 1))
            spikes = randy < (tmp_d * fr)
            # get logits
            outs,_ = snn_network.Test(spikes.astype(float),1)
            logits = outs/T

            pred1[i * batch_size:(i + 1) * batch_size], energies[i] = snn_network.Train(spikes.astype(float), (
            label[s_index[i * batch_size:(i + 1) * batch_size]]), batch_size, logits= logits)
        acn = sum(pred1 == label[s_index[:train_size]]) / float(train_size)



train_oe()
np.save("oe_w_h.npy", snn_network.w_h)
np.save("oe_w_o.npy", snn_network.w_o)











def test_snn(snn_network,d_30_test, d_30_test_label, dense_size = 32*32, conv= False, T =100 , test_batch =1):
    
    in_score = []
    right_score = []
    wrong_score = []
    
    fr =1
    if conv:
       print("wrong")
    total_test = len(d_30_test_label)
    out = []
    pred = np.zeros([total_test])
    for i2 in trange(int (total_test/test_batch)):
        tmp_rand = np.random.random([T, 1, 1])
        randy = np.tile(tmp_rand, (1, test_batch, dense_size))
        tmp_d = np.tile(d_30_test[:, i2 * test_batch:(i2 + 1) * test_batch, :], (T, 1, 1))
        spikes2 = randy < (tmp_d * fr)
        outs, pred[i2 * test_batch:(i2 + 1) * test_batch] = snn_network.Test(spikes2.astype(float), test_batch)
        if len(out) ==0:
            out = outs
        else:
            out = np.vstack((out,outs))
    
    acn = sum(pred == d_30_test_label[:total_test]) / float(total_test)
    print("final test result is : ", acn)
    wrong_index = np.where(pred !=d_30_test_label[:total_test] )
    correct_index = np.where(pred ==d_30_test_label[:total_test] )

    return acn , out,correct_index,wrong_index

acc, outs, r_index, w_index = test_snn(snn_network, dataTest,labelTest)
outs = outs/T
in_score = softmax(outs, axis= -1)
in_score =outs
in_score = -np.max(in_score, axis=1)

print(in_score)




def get_ood_scores(snn_network,d_30_test, d_30_test_label, dense_size = 32*32, conv= False, T =100 , test_batch =1):
    
    in_score = []

    
    fr =1
    if conv:
       print("wrong")
    total_test = len(d_30_test_label)
    out = []
    pred = np.zeros([total_test])
    for i2 in trange(int (total_test/test_batch)):
        tmp_rand = np.random.random([T, 1, 1])
        randy = np.tile(tmp_rand, (1, test_batch, dense_size))
        tmp_d = np.tile(d_30_test[:, i2 * test_batch:(i2 + 1) * test_batch, :], (T, 1, 1))
        spikes2 = randy < (tmp_d * fr)
        outs, pred[i2 * test_batch:(i2 + 1) * test_batch] = snn_network.Test(spikes2.astype(float), test_batch)
        if len(out) ==0:
            out = outs
        else:
            out = np.vstack((out,outs))
    

    outs = out/T
    in_score = softmax(outs, axis= -1)
    in_score = outs
    in_score = -np.max(in_score, axis=1)

    return in_score


out_score = get_ood_scores(snn_network,ood_data, ood_label, dense_size = 32*32, conv= False, T =100 , test_batch = 1)

auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(in_score, out_score, num_to_avg= 1):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
    print_measures(auroc, aupr, fpr, "mstar")

get_and_print_results(in_score, out_score)