#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: autoencoder.py |
Created on the 2019-02-22 |
Github: https://github.com/pl19n72019

This file contains the network.AutoEncoder class which provides useful
functions to init, train, load, and save autoencoder models.
"""



import glob
import os
import numpy as np

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from src.networks.generator import GeneratorAutoencoder


class AutoEncoder:
    """Global functionalities of the auto-encoders.

    This class treats the encoder and the decoder parts. It allows to train
    auto-encoders (define in the package `models`), and to perform all classic
    operations on the auto-encoders.
    """

    def __init__(self, model=None, dataset_path=None, batch_size=128,
                 load_models=None, version=0):
        """Creation of the auto-encoder functionalities treatment object.

        It loads and create the dataset and the models. Only 80% of the load
        data are used for the training phase. The other 20% are use to validate
        the model during the fitting phase. Moreover, the shape of the data
        (train and test) are fitting to run on 2-dimensional convolutional
        neural network (which offers the best results). If other network are
        used, reshape the data at the input of the network. A generator
        approach is used for loading the data, which means that the data is
        loaded batch per batch.

        Args:
            model: Model of auto-encoder (should be created in the package
                `models`) (default: None). Set it to None if you're loading an
                existing model.
            dataset_path (str): Path to the dataset (default: None). If it is
                set to None, the current path is selected. Note that you need to
                point toward a dataset folder as described in the maintenance section
                of the documentation.
            batch_size (int): Size of the batch (default: 128).
            load_models (str): Model to load (default: None). If it is set to
                None, no model is loaded. In that case model should not be set
                at None.
            version (int): Version of encoder to load (default: 0). If
                load_models is not None, it refers to the version of encoder to
                load. By default, it load the first model created by the outer-
                loop.

        Note:
            This file should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        self.batch_size = batch_size

        if load_models is None:
            # load the model (complete auto-encoder) and the encoder
            self.__model = model
            self.__encoder = self.__model.encoder()
            self.__autoencoder = self.__model.autoencoder()

            # creation of the generators
            files = glob.glob('{}/train_TS/*.npy'.format(
                './dataset' if dataset_path is None else dataset_path))
            ts_train, ts_test, _, _ = train_test_split(files, files,
                                                       test_size=0.2)
            self._nb_train_samp = len(ts_train)
            self._nb_test_samp = len(ts_test)

            self._generator_train = GeneratorAutoencoder(ts_train,
                                                         self.batch_size)
            self._generator_test = GeneratorAutoencoder(ts_test,
                                                        self.batch_size)
        else:
            with open('./saves/architectures/{}.json'.format(load_models),
                      'r') as architecture:
                pp_model = model_from_json(architecture.read())
            self.__encoder = pp_model
            self.load_weights(load_models, full=False, version=version)

    def compile(self, optimizer='adadelta', loss='mean_squared_error'):
        """Compile the complete model.

        This method should be call only on new auto-encoders (the flag
        load_models should be set to None).

        Args:
            optimizer (str (name of optimizer) or optimizer instance):
                Optimizer used to train the neural network (default:
                'adadelta').
            loss (str (name of objective function) ob objective function):
                Objective function used to analyze the network during the
                different phases (default: 'mean_squared_error').


        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        self.__autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, epochs=50, repeat=1, fname='autoencoder',
            fname_enc='encoder', verbose=1):
        """Trains the model for a given number of epochs (iterations on a
        dataset).

        This method should be call only on new auto-encoders (the flag
        load_models should be set to None).

        Args:
            epochs (int): Number of epochs to run on each outer-loop (default:
                50). At the end of each epoch iterations, the neural networks
                are saved.
            repeat (int): Number of outer-loops (default: 1). At the end of the
                training phase, `epochs * repeats` epochs are performed.
                Moreover, `repeat` networks are saved on the disk.
            fname (str): Name of the complete neural network on the disk
                (default: 'autoencoder').
            fname_enc (str): Name of the encoder part of the neural network on
                the disk (default: 'encoder').
            verbose (int, 0, 1, or 2): Verbosity mode (default: 1). 0 = silent,
                1 = progress bar, 2 = one line per epoch.

        Returns:
            A record of training loss values and metrics values at successive
            epochs, as well as validation loss values and validation metrics
            values (if applicable).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        # saving the two architectures
        self.save_architecture(fname=fname)
        self.save_architecture(fname=fname_enc, full=False)

        # training of the model and saving of the history
        history = []
        for i in range(repeat):
            h = self.__autoencoder. \
                fit_generator(generator=self._generator_train,
                              steps_per_epoch=(self._nb_train_samp// \
                                               self.batch_size),
                              epochs=epochs,
                              verbose=verbose,
                              validation_data=self._generator_test,
                              validation_steps=(self._nb_test_samp// \
                                                self.batch_size))
            history.append(h.history)
            self.save_weights('{}.{}'.format(fname, i), architecture=False)
            self.save_weights('{}.{}'.format(fname_enc, i), full=False,
                              architecture=False)

        # the format of history is flatten
        keys = history[0].keys()
        return {k: np.array([l[k] for l in history]).flatten() for k in keys}

    def predict(self, x, batch_size=128):
        """"Generates complete output predictions for the input samples
        (outputs of the auto-encoder neural network).

        This method should be call only on new auto-encoders (the flag
        load_models should be set to None).

        Args:
            x (numpy.ndarray like): Input data.
            batch_size (int): Number of samples per pass (default: 128).

        Returns:
            Numpy array(s) of reshaped predictions (for displaying).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        _, width, height, _ = self.__encoder.output_shape
        return self.__encoder.predict(x, batch_size=batch_size) \
            .reshape((len(x), width, height))

    def encode(self, x, batch_size=128):
        """"Generates output predictions for the input samples (outputs of the
        encoder neural network).

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Args:
            x (numpy.ndarray like): Input data. Should be an array of shape
                (nb_images, width_im, height_im, 1). 1 stands for grayscale
                images.
            batch_size (int): Number of samples per pass (default: 128).

        Returns:
            Numpy array(s) of reshaped predictions (for two-dimensional
            convolutional neural network).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        _, width, height, _ = self.__encoder.output_shape
        return self.__encoder.predict(x, batch_size=batch_size) \
            .reshape((len(x), width, height, 1))

    def encode_dataset(self, data_in, data_out):
        """Generate an encoded data from a timestack data.

        Be aware of the fact that your data needs to have the same
        dimension that the one required by the model. Be aware of the
        fact that your different path have to point toward a dataset folder
        as defined in maintenance section of the documentation.

        Args:
            data_in (str): Path to the data to encode (put '.' if you want
                to specify that the data is in the local path).
            data_out (str): Path where to stock the encoded data
                (put '.' if you want to specify that the data is in
                the local path).
        """
        _, width, height, _ = self.__encoder.input_shape
        ts_names = glob.glob(data_in + '/train_TS/*.npy')
        for data in ts_names:
            ts = np.array([np.load(data)]) #[200:]])
            ts_encoded = self.__encoder.predict(ts.reshape(1, width, height,
                                                           1))[0,:,:,0]
            print(data[-12:])
            np.save(data_out + '/train_encoded_TS/' + data[-12:],
                    ts_encoded)


    def save_weights(self, fname='autoencoder', full=True, architecture=True):
        """Saves the weights of the networks.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None) if full is set to False. If not, this
        method should not be call.

        Args:
            fname (str): Name of the file to save (default: 'autoencoder').
            full (bool): Network flag (default: True). If `full` is set to
                True, the auto-encoder is saved and if not, only the encoder is
                saved.
            architecture (bool): Architecture flag (default: True). If
                `architecture` is set to True, the architecture the the network
                related to the flag `full` is saved in a JSON format.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        file = './saves/weights/{}.h5'.format(fname)
        if full:
            self.__autoencoder.save_weights(file)
        else:
            self.__encoder.save_weights(file)

        if architecture:
            self.save_architecture(fname, full=full)

    def save_architecture(self, fname='autoencoder', full=True):
        """Saves the architectures of the networks.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None) if full is set to False. If not, this
        method should not be call.

        Args:
            fname (str): Name of the file to save (default: 'autoencoder').
            full (bool): Network flag (default: True). If `full` is set to
                True, the auto-encoder is saved and if not, only the encoder is
                saved.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        file = './saves/architectures/{}.json'.format(fname)
        if full:
            with open(file, 'w') as architecture:
                architecture.write(self.__autoencoder.to_json())
        else:
            with open(file, 'w') as architecture:
                architecture.write(self.__encoder.to_json())

    def load_weights(self, fname='autoencoder', full=True, version=0):
        """Loads the weights of the networks.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None) if full is set to False. If not, this
        method should not be call.

        Args:
            fname (str): Name of the file to save (default: 'autoencoder').
            full (bool): Network flag (default: True). If `full` is set to
                True, the auto-encoder is saved and if not, only the encoder is
                saved.
            version (int or str): Version of the network weights to load
                (default: 0). It refers to the `repeat` flag in the fitting
                method.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        file = './saves/weights/{}.{}.h5'.format(fname, version)
        if full:
            self.__autoencoder.load_weights(file)
        else:
            self.__encoder.load_weights(file)

    def save_losses(self, history, fname='autoencoder'):
        """Saves the history given as input.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Args:
            history (dict): History the save on disk.
            fname (str): Name of the file to save (default: 'autoencoder').

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        np.save('./saves/losses/{}.npy'.format(fname),
                history)

    def load_losses(self, fname='autoencoder'):
        """Loads the history given as input.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Args:
            fname (str): Name of the file to load (default: 'autoencoder').

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.
        """
        return np.load(
            './saves/losses/{}.npy'.format(fname)).item()
