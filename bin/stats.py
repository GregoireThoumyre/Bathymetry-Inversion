#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################
# Save data with stats of prediction on 1000 cases                   #
# prediction with cnn, mlp, greg's methods                           #
######################################################################


import context
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.io as sio
from math import sqrt
from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder
import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense




####### CNN model ###########
model = 'Model2_test15'
version_model = 0
cnn_name = 'cnn-'
# load an already trained cnn model
ae1 = AutoEncoder(load_models=('encoder-' + 'Model1_test15'), version=0)
cnn1 = CNN(load_models=(cnn_name + model), version=version_model)

####### MLP model ###########
name = 'mlp'
metric_name = 'loss'
mlp = Sequential()
mlp.add(Dense(1620, kernel_initializer='normal', activation='relu', input_dim=405))
mlp.add(Dense(810, kernel_initializer='normal', activation='relu'))
mlp.add(Dense(405, kernel_initializer='normal', input_dim=405))
mlp.load_weights('./saves/weights/{}.h5'.format(name))

####### Load greg predictions
M = sio.loadmat('../../Shifts.mat')
hcss = M['h_css']

####### Stat loop ###########
rmse1 = []
rmse2 = []
rmse3 = []
me1 = []
me2 = []
me3 = []
n_init = 15321
for i in range(1000):
    n = i + n_init
    real_bathy = np.load('dataset_celerity_val/train_GT/B_' + str(n) + '.npy')
    cele = np.load('dataset_celerity_val/train_TS/cele/C_' + str(n) + '.npy')
    bathy_pred_greg = np.array(hcss[:, n - 1])


    ts_origi = np.load('../../dataset_new/train_TS/TS_' + str(n) + '.npy')
    width, height = ts_origi.shape
    ts_origi = np.array([ts_origi])
    ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
    a, width, height = ts_enc.shape
    ts_enc = np.array([ts_enc])
    pred_bathy_ts = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)


    x = np.arange(0, 405, 1)  # cross-shore
    # x = x.reshape((1, 405))
    pred_bathy_cele = mlp.predict(cele)
    pred_bathy_cele = pred_bathy_cele.reshape((405,))
    pred_bathy_ts = pred_bathy_ts[0,75:480]
    bathy_pred_greg = -bathy_pred_greg[::-1]
    real_bathy = real_bathy[75:480]
    real_bathy = real_bathy.reshape((405,))

    rmse1.append(mean_squared_error(real_bathy, pred_bathy_ts))
    me1.append(np.mean(np.abs(real_bathy - pred_bathy_ts)))
    # print(rmse1)
    if any(np.isnan(pred_bathy_cele)):
        print('Nan inside pred_bathy_cele')
        print(i)
    else :
        rmse2.append(mean_squared_error(real_bathy, pred_bathy_cele))
        me2.append(np.mean(np.abs(real_bathy - pred_bathy_cele)))
        # print(rmse2)
    if any(np.isnan(bathy_pred_greg)):
        print('Nan inside bathy_pred_greg')
        print(i)
    else :
        rmse3.append(mean_squared_error(real_bathy, bathy_pred_greg))
        me3.append(np.mean(np.abs(real_bathy - bathy_pred_greg)))
        # print(rmse3)
    # print('############################')

np.save('me1.npy', me1)
np.save('me2.npy', me2)
np.save('me3.npy', me3)
np.save('rmse1.npy', rmse1)
np.save('rmse2.npy', rmse2)
np.save('rmse3.npy', rmse3)

print(rmse1.__len__())
print(rmse2.__len__())
print(rmse3.__len__())

    # else :
    #     cpt += 1
    #     print(cele)
    #     print(cpt)
    #     print('NaN inside greg prediction')

