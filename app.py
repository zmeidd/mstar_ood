import io
import base64
from pyexpat import model
from numpy import average

import torch
import torchvision.transforms as trn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from models.wrn import WideResNet
from flask import request
from flask import Flask, jsonify, request, redirect, render_template


## EMSTDP Module
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
# None OE Model:
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
# train_size = len(train_data)
# test_size = len(test_data)
# energies = np.zeros([train_size])
# batch_energy = np.zeros([int(train_size / 50)])  # bach_size = 50
ind += 1
acc = []
np.random.seed(0)
tmp_rand = np.random.random([T, 1, 1])
randy = np.tile(tmp_rand, (1, batch_size, dense_size))
tmp_d = np.zeros([T, batch_size, dense_size])

lr = 0.003
online_norm = False
####### parameter ends ############

'''
None OE EMSTDP
'''
snn_network = svp.Network(0, dropr, 0, 0.0, rel, delt, 1, 0, clp, lim, dense_size, h, 2, hiddenThr1*fr, outputThr1*fr, T, bias, lr, scale, twin, epsilon,

                         online_norm= False, train_oe= True)
w_h = np.load("w_h.npy")
w_o = np.load("w_o.npy")
snn_network.w_h = w_h
snn_network.w_o = w_o


'''
OE SNN 
'''
snn_oe = svp.Network(0, dropr, 0, 0.0, rel, delt, 1, 0, clp, lim, dense_size, h, 2, hiddenThr1*fr, outputThr1*fr, T, bias, lr, scale, twin, epsilon,

                         online_norm= False, train_oe= True)
snn_oe.w_h = np.load("oe_w_h.npy")
snn_oe.w_o = np.load("oe_w_o.npy")


'''
SNN Scores
'''
####

def get_ood_scores(snn_network,image_bytes, dense_size = 32*32, conv= False, T =100 , test_batch =1):
    np.random.seed(0)
    tensor = transform_image(image_bytes=image_bytes)
    d_30_test = tensor.cpu().numpy()
    d_30_test = np.reshape(d_30_test, (1,len(d_30_test),dense_size)).astype(float)
    in_score = []
    fr =1
    if conv:
       print("wrong")
    total_test = len(d_30_test)
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

#### get ood predictions #########
def emstdp_prediction(img_bytes, model_param):
    score = []
    if(model_param[0] == 'option1'):
        # for i in range(20):
        #     score_ = get_ood_scores(snn_network= snn_oe, image_bytes= img_bytes)
        #     score.append(score_)
        score = get_ood_scores(snn_network= snn_oe, image_bytes= img_bytes)
    else:
        # for i in range(20):
        #     score_ = get_ood_scores(snn_network= snn_network, image_bytes= img_bytes)
        #     score.append(score_)
        score = get_ood_scores(snn_network= snn_network, image_bytes= img_bytes)

    return score


app = Flask(__name__)

# layers = 22
# num_classes = 2
# widen_factor = 2
# droprate = 0.3
# net1 = WideResNet(layers, num_classes, widen_factor, dropRate=droprate)
# net2 = WideResNet(layers, num_classes, widen_factor, dropRate=droprate)
# model1_name = "/home/boyu/outlier-exposure/CIFAR/snapshots/mstar/mstar_wrnoe_mstar32_newtask2_epoch_212.pt"
# model2_name = "/home/boyu/outlier-exposure/CIFAR/snapshots/mstar/mstar_wrnmstar32_newtask2_epoch_199.pt"
# net1.load_state_dict(torch.load(model1_name,map_location='cpu'))
# net2.load_state_dict(torch.load(model2_name,map_location='cpu'))
# net1.eval()
# net2.eval()
def transform_image(image_bytes):
    temp_transform = trn.Compose([trn.Resize([32,32]),trn.ToTensor()])

    image = Image.open(io.BytesIO(image_bytes))

    return temp_transform(image).unsqueeze(0)



# def get_prediction(image_bytes, model_param):
#     tensor = transform_image(image_bytes=image_bytes)
#     print(model_param)
#     if(model_param[0] == 'option1'):
#         print("load model1")
#         with torch.no_grad():
#             output = net1(tensor[:, :1, :, :])
        
#     else:
#         print("load model2")
#         with torch.no_grad():
#             output = net2(tensor[:, :1, :, :])

#     to_np = lambda x: x.data.cpu().numpy()

#     smax = to_np(F.softmax(output, dim=1))
#     print(smax)
#     maxscore = np.max(smax, axis=1)
#     print(maxscore)
#     return maxscore



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        EMSTDP = True
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        choose_model = request.form.getlist('options')
        if not file:
            return
        img_bytes = file.read()

        encoded_img_data = base64.b64encode(img_bytes)
        if EMSTDP:
            predication_score = emstdp_prediction(img_bytes = img_bytes, model_param= choose_model)
            print("EMSTDP prediction score: ", predication_score)
        # else: 
        #     predication_score = get_prediction(image_bytes=img_bytes, model_param = choose_model)
        return render_template('result.html', score = predication_score , img = encoded_img_data.decode('utf-8'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)

