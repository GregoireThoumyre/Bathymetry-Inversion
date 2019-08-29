#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################
# Train a new Auto-Encoder and save the encoder                      #
#                                                                    #
######################################################################

import context
import os
from src.models import autoencoder
from src.networks.autoencoder import AutoEncoder


if __name__ == '__main__':
    dataset_path = '../GREGOIRE/dataset_new' # '.'
    name = 'Model1_test17'
    # creation and training of an auto-encoder, to pre-process the data
    ae1 = AutoEncoder(model=autoencoder.Model1((520, 480, 1)), batch_size=64, dataset_path=dataset_path) # Adapt to your path
    ae1.compile()
    hae1 = ae1.fit(epochs=2, repeat=1, fname=('autoencoder-'+name), fname_enc=('encoder-'+name))
    ae1.save_losses(hae1, 'encoder-'+name) # saving the losses

    # Create the encoded dataset with the encoder
    #ae1 = AutoEncoder(load_models='encoder-Model1_test5', version=0)
    if not os.path.exists(dataset_path + '/train_encoded_TS/'):
        os.mkdir(dataset_path + '/train_encoded_TS/')
    ae1.encode_dataset(dataset_path, dataset_path)
