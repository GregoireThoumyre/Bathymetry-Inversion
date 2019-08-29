#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import context
import numpy as np
import matplotlib.pyplot as plt

from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder

if __name__ == '__main__':
    # load an already trained cnn model
    ae1 = AutoEncoder(load_models='encoder-Model1_test10', version=0)
    cnn1 = CNN(load_models='cnn-Model1_test10', version=9)


    # predict cnn, 20 predict
    ts_names = sorted(glob.glob('../dataset_valid/train_TS/*.npy'))
    bathy_names = sorted(glob.glob('../dataset_valid/train_GT/*.npy'))

    x = np.arange(0, 480, 1)  # cross-shore

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        
        ts_origi = np.load(ts_names[i])#[200:]	# croping
        width, height = ts_origi.shape
        ts_origi = np.array([ts_origi])
        real_bathy = np.load(bathy_names[i])
        
        ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
        a ,width, height = ts_enc.shape
        ts_enc = np.array([ts_enc])
        
        pred_bathy = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
        plt.plot(x, real_bathy, color='xkcd:chocolate', label='Expected', linewidth=2.0)
        plt.plot(x, pred_bathy.flatten(), color='red', label='Prediction', linewidth=2.0)
        plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
        plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')
        # plt.plot(real_bathy, label='Expected')
        # plt.plot(pred_bathy.flatten(), label='Prediction')
        plt.legend()
    plt.show()
