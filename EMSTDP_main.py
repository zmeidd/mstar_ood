import os
from cv2 import log

from scipy.ndimage.measurements import labeled_comprehension
from scipy.special import softmax
from matplotlib import pyplot as plt

import sys
# import pyximport; pyximport.install()

import numpy as np
import EMSTDP_algo as svp
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
from utils import train_snn,test_snn,train_oe
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
dense_size = 28*28
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
snn_network = svp.Network(0, dropr, 0, 0.0, rel, delt, 1, 0, clp, lim, dense_size, [100,100], 6, hiddenThr1*fr, outputThr1*fr, T, bias, lr, scale, twin, epsilon,
online_norm= True)

train_acc =[]
test_acc = []
## load trained weights, load_weight=True
load_weight = False
load_epoch = 1

ver_period = 1000
s_index = data_index

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
# data_ood = np.zeros((len(data_oe),32,32))
path = os.getcwd() +"/incre_data/"




'''
take the random indices and return dataset
'''

def random_idx(train_,train_label, oe_train, oe_label):
    idx = np.arange(train_.shape[0]+oe_train.shape[0])
    np.random.shuffle(idx)
    data = np.vstack((train_,oe_train))
    label = np.concatenate((train_label, oe_label))
    data = data[idx]
    label = label[idx]
    
    return data, label
def ood_data():
        d_1 = np.load(path+"state_"+str(0)+"_x.npy")
        d_1 = 255*np.reshape(d_1,(-1,28,28)).astype(float)
        d_2 = np.load(path+"state_"+str(1)+"_x.npy")
        d_2 = 255*np.reshape(d_2,(-1,28,28)).astype(float)
        ood_data = np.vstack((d_1,d_2))
        d1_label = np.load(path+"state_"+str(1)+"_y.npy")
        d2_label = np.load(path+"state_"+str(2)+"_y.npy")
        ood_label = np.concatenate((d1_label,d2_label))
        
        return d_1,d1_label,d_2,d2_label


# train 1000 outlier samples
def outlier_exposure(snn_network = snn_network):
    label = np.random.randint(6,size= (2000,))
    data = TRAINING[0][:2000]
    data = np.reshape(data, (len(data),28,28))
    scale = 1/np.max(data)
    data = scale*data
    data = np.expand_dims(np.reshape(data, [-1,dense_size]), axis=0)
    
    snn_network = train_oe(snn_network,data,label, batch_size= 100
                                ,dense_size= 784,detect_size= 1)
    
    return snn_network
    

def test_2_ood():
        # d_1 = np.load(path+"state_"+str(1)+"_x.npy")
        # d_1 = 255*np.reshape(d_15_train,(-1,28,28)).astype(float)[:100]
        d_2 = np.load(path+"state_"+str(2)+"_x.npy")
        d_2 = 255*np.reshape(d_2,(-1,28,28)).astype(float)[:200]
        # ood_data = np.vstack((d_1,d_2))
        # d1_label = np.load(path+"state_"+str(1)+"_y.npy")[:100]
        d2_label = np.load(path+"state_"+str(2)+"_y.npy")[:200]
        # ood_label = np.concatenate((d1_label,d2_label))
        
        return d_2, d2_label  
    
def test_1_ood():
        d_1 = np.load(path+"state_"+str(1)+"_x.npy")
        d_1 = 255*np.reshape(d_15_train,(-1,28,28)).astype(float)[:100]
        d_2 = np.load(path+"state_"+str(2)+"_x.npy")
        d_2 = 255*np.reshape(d_2,(-1,28,28)).astype(float)[:100]
        ood_data = np.vstack((d_1,d_2))
        d1_label = np.load(path+"state_"+str(1)+"_y.npy")[:100]
        d2_label = np.load(path+"state_"+str(2)+"_y.npy")[:100]
        ood_label = np.concatenate((d1_label,d2_label))
        
        return ood_data, ood_label 


    
    
    


state_size = 3
def TPR95(score):
    count = int(len(score)*0.05)
    idx = np.argsort(score)
    return score[idx[count]]

for i in range(1):
        print("state number: " +str(i))
        # d_15_train = np.load(path+"state_"+str(i)+"_x.npy")
        data_oe = TRAINING[0][0:200*(state_size+1)]
        data_oe = np.reshape(data_oe, (len(data_oe),28,28))
        oe_label = TRAINING[1][0:200*(state_size+1)]
        d_15_train = np.load(path+"state_"+str(i)+"_x.npy")
        d_15_train = 255*np.reshape(d_15_train,(-1,28,28)).astype(float)
        scale = 1/np.max(d_15_train)
        d_15_train = scale*d_15_train[:400]
        d_15_test = d_15_train
        h,w = d_15_train.shape[1],d_15_train.shape[2]
        d_15_train_label = np.load(path+"state_"+str(i)+"_y.npy")[:400]
        # data,label = random_idx(d_15_train,d_15_train_label,d_15_train,d_15_train_label)
        d_15_train_new = np.expand_dims(np.reshape(d_15_train, [-1,dense_size]), axis=0)
        for nn in range(3):
                snn_network = train_snn(snn_network,d_15_train_new,d_15_train_label, batch_size= 10
                                        ,dense_size= 784,detect_size= 1)
                if nn%2 == 0:
                    outlier_exposure()
        acc,outs,_,_ =  test_snn(snn_network,d_15_train_new,d_15_train_label, dense_size= 784)
        
        
        
        in_score = np.max(outs, axis=1)
        print(in_score[:20])
        threshold = TPR95(in_score)
        indx = np.argsort(in_score)
        print("minimum value", in_score[indx[:10]])
       
        
        ood_data,ood_label = test_2_ood()
        ood_new = np.expand_dims(np.reshape(ood_data, [-1,dense_size]), axis=0)
    
        print("ood_score length: ", len(ood_label))
    
        acc,outs,_,_ = test_snn(snn_network,ood_new,ood_label, dense_size= 784)
        out_score = np.max(outs, axis=1)
        print("out scores: ", out_score[:20])
        
        '''
        Calculate FP
        '''
        FP = 0
        TN = 0
        for kk in range(len(out_score)):
            if out_score[kk]> threshold:
                FP+=1
        print("out score length: ", len(out_score))
        print(FP)

def oe_task_2():
    for i in range(1):
            print("state number: " +str(i))
            # d_15_train = np.load(path+"state_"+str(i)+"_x.npy")
            # d_15_train = np.load(path+"state_"+str(i)+"_x.npy")
            # d_15_train = 255*np.reshape(d_15_train,(-1,28,28)).astype(float)
            d_1,d1_label,d_2,d2_label = ood_data()
            d_15_train,d_15_train_label = random_idx(d_1,d1_label, d_2,d2_label)
            scale = 1/np.max(d_15_train)
            d_15_train = scale*d_15_train[:400]
            d_15_test = d_15_train
            h,w = d_15_train.shape[1],d_15_train.shape[2]
        
            d_15_train_label = d_15_train_label[:400]
        
            d_15_train_new = np.expand_dims(np.reshape(d_15_train, [-1,dense_size]), axis=0)
            for nn in range(40):
                snn_network = train_snn(snn_network,d_15_train_new,d_15_train_label, batch_size= 10
                                        ,dense_size= 784,detect_size= 1)
                if nn%2 == 0:
                    outlier_exposure()
                
            acc,outs,_,_ =  test_snn(snn_network,d_15_train_new,d_15_train_label, dense_size= 784)
            ood_data,ood_label = test_2_ood()
            scale = 1/np.max(ood_data)
            ood_data = scale*ood_data
            ood_new = np.expand_dims(np.reshape(ood_data, [-1,dense_size]), axis=0)
        
        
            # outs = outs/T
            # in_score = softmax(outs, axis= -1)
            in_score = np.max(outs, axis=1)
            in_score = in_score[:400]
            th = TPR95(in_score)
            print(in_score[:10])
            print(th)
        
            acc,outs,_,_ = test_snn(snn_network,ood_new,ood_label, dense_size= 784)
            print("out score length, ", outs )
            # outs = outs/T
            # out_score = softmax(outs, axis= -1)
            out_score = np.max(outs, axis=1)
            print(out_score[:10])
            '''
            Calculate FP
            '''
            FP = 0
            TN = 0
            for kk in range(len(out_score)):
                if out_score[kk]> th:
                    FP+=1
            print(len(out_score))
            print(FP)
