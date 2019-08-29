#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################
# Plot 1 figure with the original TS, the encoded TS and the decoded #
# TS                                                                 #
######################################################################

import context
import numpy as np
import matplotlib.pyplot as plt

from src.models import autoencoder
from src.networks.autoencoder import AutoEncoder


if __name__ == '__main__':
    # load an already trained auto-encoder model
    ae1 = AutoEncoder(load_models='encoder-Model1_test15', version=0)


    # predict encoder
    ts_origi = np.load('../../dataset_new/train_TS/TS_00003.npy') #[200:] # adapt you path
    width, height = ts_origi.shape
    ts_origi = np.array([ts_origi], dtype='float32')
    ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
    print(ts_origi.dtype)
    print(ts_enc.dtype)
    tsori = ts_origi[0,:,:]
    tsenc = np.array(ts_enc[0,:,:])
    plt.subplot(1, 3, 1)
    plt.imshow(tsori)
    plt.title('Original Timestack')
    plt.subplot(1, 3, 2)
    plt.imshow(tsenc)
    plt.title('Encoded Timestack')
    plt.subplot(1, 3, 3)
    plt.imshow(tsori)
    plt.title('Decoded Timestack')
    plt.show()
