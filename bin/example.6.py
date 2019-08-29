#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################
# Plot 1 figure with 20 different bathy                              #
# prediction with cnn                                                #
######################################################################

import glob
import context
import numpy as np
import matplotlib.pyplot as plt

from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder

if __name__ == '__main__':

    # Variables
    dataset = 'datatest2'
    model = 'Model2_test15'
    version_model = 0

    # load an already trained cnn model
    ae1 = AutoEncoder(load_models=('encoder-' + 'Model1_test15'), version=0)
    cnn1 = CNN(load_models=('cnn-' + model), version=version_model)

    # predict cnn, 20 predict
    ts_names = sorted(glob.glob(dataset + '/train_TS/*.npy'))
    bathy_names = sorted(glob.glob(dataset + '/train_GT/*.npy'))

    x = np.arange(0, 480, 1)  # cross-shore

    shift = 40
    for i in range(20):
        plt.subplot(4, 5, i + 1)

        ts_origi = np.load(ts_names[i+shift])[200:720, :]  # croping
        width, height = ts_origi.shape
        ts_origi = np.array([ts_origi])
        real_bathy = np.load(bathy_names[i+shift])
        # ts_mean = np.mean(ts_origi)
        # ts_var = np.var(ts_origi, 0)
        # ts_origi = (ts_origi - ts_mean) / (ts_var ** 2)

        ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
        a, width, height = ts_enc.shape
        ts_enc = np.array([ts_enc])

        pred_bathy = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
        plt.plot(x, real_bathy, label='Expected', color='k')
        plt.plot(x, pred_bathy.flatten(), label='Prediction', color='r')
        plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
        plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')
    plt.legend()
    plt.show()
