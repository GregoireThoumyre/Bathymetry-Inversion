#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import context
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder


if __name__ == '__main__':
    print('Start')

    mbathy = sio.loadmat('../../../Greg/Bathy_in_situ/Bathy_Prof_GPP2014.mat')
    xbathy = mbathy['x']
    zbathy = mbathy['z']
    real_bathy = np.array(zbathy[0, 61:541])
    # print(np.shape(np.array(zbathy))) :(1, 641)
    # plt.plot(zbathy[0])

    ae1 = AutoEncoder(load_models='encoder-Model1_test17', version=0)
    cnn1 = CNN(load_models='cnn-Model2_test17', version=0)

    pred_bathys = []
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 2, 1)
    # ax1.set_title('All predictions from all TS')
    for name in os.listdir('others/real_data/'):
        print(name)

        mts1 = sio.loadmat('others/real_data/'+name)
        ts1 = mts1['ts']
        # print(np.shape(np.array(ts1))) : (1806, 718)
        # plt.imshow(ts1)

        ts_origi = ts1[0:520, 0:480] #238:718]
        width, height = ts_origi.shape
        ts_origi = np.array([ts_origi])
        ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
        a, width, height = ts_enc.shape
        ts_enc = np.array([ts_enc])
        pred_bathy_ts = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
        pred_bathys.append(pred_bathy_ts[0])

        x = np.arange(0, 480, 1)  # cross-shore vector
        plt.plot(x, real_bathy, label='Expected', color='k')
        plt.plot(x, pred_bathy_ts[0], label='Pred_ts')
        plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
        plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.set_title('Mean bathy')
    pred_bathy_mean = np.mean(pred_bathys, 0)
    plt.plot(x, real_bathy, label='Expected', color='k')
    plt.plot(x, pred_bathy_mean, label='Pred_ts')
    plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
    plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')
    # plt.legend()
    plt.show()

    print('End')

