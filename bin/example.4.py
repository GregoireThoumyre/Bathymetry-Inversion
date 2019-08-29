#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context
import numpy as np
import matplotlib.pyplot as plt

from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder


if __name__ == '__main__':
    # Variables
    dataset = '../../dataset_new'
    model = 'Model2_test17'
    model_encoder = 'Model1_test15'

    # load an already trained cnn model
    ae1 = AutoEncoder(load_models=('encoder-' + model_encoder), version=0)
    cnn0 = CNN(load_models=('cnn-' + model), version=0)
    # cnn1 = CNN(load_models=('cnn-' + model), version=1)
    # cnn2 = CNN(load_models=('cnn-' + model), version=2)
    # cnn3 = CNN(load_models=('cnn-' + model), version=3)
    # cnn4 = CNN(load_models=('cnn-' + model), version=4)
    # cnn5 = CNN(load_models=('cnn-' + model), version=5)
    # cnn6 = CNN(load_models=('cnn-' + model), version=6)
    # cnn7 = CNN(load_models=('cnn-' + model), version=7)
    # cnn8 = CNN(load_models=('cnn-' + model), version=8)
    # cnn9 = CNN(load_models=('cnn-' + model), version=9)

    x = np.arange(0, 480, 1)  # cross-shore

    # predict cnn
    n = '15353'
    ts_origi = np.load(dataset+'/train_TS/TS_'+n+'.npy')  # [200:]	# croping
    width, height = ts_origi.shape
    ts_origi = np.array([ts_origi])
    print(ts_origi.shape)
    real_bathy = np.load(dataset+'/train_GT/B_'+n+'.npy')

    ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
    a, width, height = ts_enc.shape
    ts_enc = np.array([ts_enc])

    pred_bathy0 = cnn0.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=False)
    pred_bathy1 = cnn0.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy1 = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy2 = cnn2.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy3 = cnn3.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy4 = cnn4.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy5 = cnn5.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy6 = cnn6.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy7 = cnn7.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy8 = cnn8.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
    # pred_bathy9 = cnn9.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)

    plt.subplot(1, 2, 1)
    ts = np.array(ts_origi, dtype='float32')
    plt.plot(x, real_bathy, color='xkcd:chocolate', label='Expected', linewidth=2.0)
    plt.plot(x, pred_bathy0.flatten(), label='Prediction', linewidth=2.0)
    plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
    plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')
    # plt.imshow(ts[0])
    plt.xlabel('Cross-shore (m)')
    plt.ylabel('Time (s)')
    # plt.title('Hovmoller diagram')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, real_bathy, color='xkcd:chocolate', label='Expected', linewidth=2.0)
    plt.plot(x, pred_bathy1.flatten(), label='Smoothed Prediction', linewidth=2.0)
    # plt.plot(x, pred_bathy1.flatten(), label='Prediction 1', linewidth=2.0)
    # plt.plot(x, pred_bathy2.flatten(), label='Prediction 2', linewidth=2.0)
    # plt.plot(x, pred_bathy3.flatten(), label='Prediction 3', linewidth=2.0)
    # plt.plot(x, pred_bathy4.flatten(), label='Prediction 4', linewidth=2.0)
    # plt.plot(x, pred_bathy5.flatten(), label='Prediction 5', linewidth=2.0)
    # plt.plot(x, pred_bathy6.flatten(), label='Prediction 6', linewidth=2.0)
    # plt.plot(x, pred_bathy7.flatten(), label='Prediction 7', linewidth=2.0)
    # plt.plot(x, pred_bathy8.flatten(), label='Prediction 8', linewidth=2.0)
    # plt.plot(x, pred_bathy9.flatten(), label='Prediction 9', linewidth=2.0)

    plt.fill_between(x, 0, real_bathy, facecolor='xkcd:azure')
    plt.fill_between(x, np.min(real_bathy), real_bathy, facecolor='orange')
    plt.xlabel('Cross-shore (m)')
    plt.ylabel('Time (s)')
    # plt.title('Bathymetry')
    plt.legend()
    plt.show()
