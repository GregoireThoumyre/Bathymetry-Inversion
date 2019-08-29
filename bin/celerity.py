#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
import context

from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



if __name__ == '__main__':

    name = 'mlp'
    metric_name = 'loss'

    mlp = Sequential()
    mlp.add(Dense(1620, kernel_initializer='normal', activation='relu', input_dim=405))
    mlp.add(Dense(810, kernel_initializer='normal', activation='relu'))
    mlp.add(Dense(405, kernel_initializer='normal', input_dim=405))

    mlp.load_weights('./saves/weights/{}.h5'.format(name))

    # save the model keras vizualisation
    # plot_model(mlp, 'mlp.png')
#############################################################
    # LOSS graph
    # history = np.load('./saves/losses/{}.npy'.format(name), allow_pickle=True).item()
    #
    # print(history.history.keys())
    # history = history.history
    # metric = history[metric_name]
    # val_metric = history['val_' + metric_name]
    # print(np.shape(metric)[0])
    # e = range(1, NB_EPOCHS + 1)

    # plt.figure(1)
    # plt.plot(metric, 'bo', label='Train ' + metric_name)
    # plt.plot(val_metric, 'b', label='Validation ' + metric_name)
    # plt.xlabel('Epoch number')
    # plt.ylabel(metric_name)
    # plt.title('Comparing training and validation ' + metric_name + ' for ')
    # plt.legend()
#############################################################

    ae1 = AutoEncoder(load_models='encoder-Model1_test17', version=0)
    cnn1 = CNN(load_models='cnn-Model2_test17', version=0)

    M = sio.loadmat('../../Shifts.mat')
    hcss = M['h_css']

    # for i in range(c.shape[1]):
    #     ci = np.array([c[:, i]])
    #     print(ci.shape)
        # name = 'train_TS/C_' + "{0:05}".format(i + 1) + '.npy'
        # np.save(name, ci)

    n_init = 15401
    plt.figure(3)
    for i in range(20):
        n = i + n_init
        real_bathy = np.load('dataset_celerity_val/train_GT/B_' + str(n) + '.npy')
        cele = np.load('dataset_celerity_val/train_TS/cele/C_' + str(n) + '.npy')
        bathy_pred_greg = np.array(hcss[:, n-1])

        ts_origi = np.load('../../dataset_new/train_TS/TS_' + str(n) + '.npy')
        width, height = ts_origi.shape
        ts_origi = np.array([ts_origi])
        ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
        a, width, height = ts_enc.shape
        ts_enc = np.array([ts_enc])
        pred_bathy_ts = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)

        # plt.figure(2)
        # plt.plot(cele[0])

        plt.subplot(4,5,i+1)
        x = np.arange(0, 405, 1)  # cross-shore
        # x = x.reshape((1, 405))
        pred_bathy_cele = mlp.predict(cele)
        pred_bathy_cele = pred_bathy_cele.reshape((405,))
        pred_bathy_ts = pred_bathy_ts[0, 75:480]
        bathy_pred_greg = -bathy_pred_greg[::-1]
        real_bathy = real_bathy[75:480]
        real_bathy = real_bathy.reshape((405,))

        # print(pred_bathy)
        plt.plot(x, real_bathy, label='Expected', color='k', linewidth=2.0)
        plt.plot(x, pred_bathy_cele, label='Pred_cele', color='r', linewidth=2.0)
        # plt.plot(x, pred_bathy_ts, label='Pred_ts', color='g')
        # plt.plot(x, bathy_pred_greg, label='Pred_cele_greg', color='b')
        plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
        plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')
    plt.legend()

    plt.show()



#     cpt = 0
#     for name in os.listdir('dataset_celerity_val/train_TS'):
#         c = np.load('dataset_celerity_val/train_TS/' + name)
#         if np.isnan(c).any():
#             print(c)
#             for i in range(405):
#                 if np.isnan(c[0,i]):
#                     c[0,i] = c[0,i-1]
#             print(c)

