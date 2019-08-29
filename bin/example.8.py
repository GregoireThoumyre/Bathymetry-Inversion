#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import context
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from netCDF4 import Dataset as netCDFFile
from mayavi import mlab
from tvtk.api import tvtk

from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder

if __name__ == '__main__':

    x = np.arange(-99, 600 - 99, 1)  # cross-shore
    y = np.arange(0, 100, 1)  # long-shore
    X, Y = np.meshgrid(x, y)

    # bathy_nc = netCDFFile('dataset_2D/dep.nc')
    # bathy = bathy_nc.variables['depth']
    # bathy = -np.array(bathy)
    bathy = np.load('../../')
    print(bathy.shape)
    ts_nc = netCDFFile('dataset_2D/eta.nc')
    ts = ts_nc.variables['eta']
    ts = np.array(ts)
    print(ts.shape)

    ae1 = AutoEncoder(load_models='encoder-Model1_test13', version=0)
    cnn1 = CNN(load_models='cnn-Model2_test14', version=0)

    bathy_pred = np.zeros([100, 480])
    for i in range(bathy.shape[0]):
        bath1D = bathy[i, 21:501]
        ts1D = ts[200:720, i, 21:501]
        width, height = ts1D.shape
        ts1D = np.array([ts1D])


        ts1D_enc = ae1.predict(ts1D.reshape(len(ts1D), width, height, 1), batch_size=1)
        a, width, height = ts1D_enc.shape
        ts1D_enc = np.array([ts1D_enc])

        bathy_pred[i, :] = cnn1.predict(ts1D_enc.reshape(len(ts1D_enc), width, height, 1), batch_size=1, smooth=True)
        print(i)

    fig = mlab.figure(size=(2000, 1000), bgcolor=(27 / 255, 133 / 255, 224 / 255))
    mlab.view(-10, -100)

    mlab.surf(x, y, bathy_pred * 10, colormap='Reds')
    mlab.surf(x, y, bathy * 10, colormap='Wistia')
    s = mlab.surf(x, y, ts[:, 0, :], colormap='winter')

    @mlab.animate(delay=100)
    def anim():
        for i in range(int(np.shape(ts)[0] - 1)):
            s.mlab_source.set(scalars=ts[i, :, :] * 50)
            yield

    anim()
    mlab.gcf().scene.parallel_projection = True
    mlab.pipeline.user_defined(bathy, filter=tvtk.CubeAxesActor())
    mlab.show()

    # # # TO .mat ##########################################################
#    DATASET = '../../dataset_new'
#    DATASETNEW = '../../dataset_new'
#    a = os.listdir(DATASET + '/train_TS/')
#    j = 0
#    print(a[1][2:9])
#    for name in os.listdir(DATASET + '/train_TS/'):
#        j += 1
#        nb = name[2:]
#        ts = np.load(DATASET + '/train_TS/' + name)
#        b = np.load(DATASET + '/train_GT/B' + nb)
#        sio.savemat(DATASETNEW + '/TSB/TSB' + nb[0:6], {'ts':ts, 'b':b})
#        print(j)




    # Variables
    # dataset = 'datatest2'
    # model = 'Model1_test12'
    # version_model = 2
    #
    # # load an already trained cnn model
    # ae1 = AutoEncoder(load_models=('encoder-' + model), version=0)
    # cnn1 = CNN(load_models=('cnn-' + model), version=version_model)
    #
    # bathy =
    # ts =
    #
    #
    # for i in range():
    #
    #     ts_origi = np.load(ts_names[i])  # [200:]	# croping
    #     width, height = ts_origi.shape
    #     ts_origi = np.array([ts_origi])
    #     real_bathy = np.load(bathy_names[i])
    #
    #     ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
    #     a, width, height = ts_enc.shape
    #     ts_enc = np.array([ts_enc])
    #
    #     pred_bathy = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)

    # PLOT TS #################################################################
    # dataset = 'datatest2'
    # # tsopti = np.load('test/TS_00003.npy.opti')
    # # ts = np.load('test/TS_00003.npy')
    # ts = np.load(dataset + '/train_TS/TS_00001.npy')
    # b = np.load(dataset + '/train_GT/B_00001.npy')
    # # ts = ts[200:720,:]
    # ts = [[float(ts[j, i]) for i in range(ts.shape[1])] for j in range(ts.shape[0])]
    # ts = np.array(ts)
    # # tsopti = [[float(tsopti[j, i]) for i in range(tsopti.shape[1])] for j in range(tsopti.shape[0])]
    # # tsopti = np.array(tsopti)
    #
    #
    # # print(ts)
    # # print(np.max(ts))
    # plt.subplot(1, 2, 1)
    # # print(ts.shape)
    # plt.imshow(ts)
    #
    # plt.subplot(1, 2, 2)
    # ts_mean = np.mean(ts)
    # ts_var = np.var(ts, 0)
    # ts_new = (ts - ts_mean)/ts_var**2
    # plt.imshow(ts_new)
    # # plt.imshow(tsopti)
    # plt.show()
#