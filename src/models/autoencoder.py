#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: autoencoder.py |
Created on the 2019-02-22 |
Github: https://github.com/pl19n72019

This file contains the different models of autoencoder.
"""



from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU
from keras.models import Sequential


class Model1:
    """A first auto-encoder model.

    The model is a convolutional model, adapted to the problem. The inputs
    have originally two dimensions, the timestack size. The shape of the output
    of the encoder is the reduce dimensions (compression).

    The encoder is composed of three convolutional layers, with non-square
    max-pooling after each convolutional layer. The decoder is classically
    composed of three convolutional layers too, in the opposite direction.

    Example:

        >>> autoencoder = Model1((400, 200, 1))
        >>> print("Encoder summary.")
        >>> print(autoencoder.encoder().summary())
        >>> print("Auto-__cnn summary.")
        >>> print(autoencoder.autoencoder().summary())
    """

    def __init__(self, input_shape):
        """Creation of a first CNN model.

        The model is a convolutional model, adapted to the problem. The inputs
        have originally two dimensions, the timestack size. The shape of the
        output of the encoder is the reduce dimensions (compression).

        Args:
            input_shape (`tuple`): Input shape of the model, typically
                (_, _, 1).

        Note:
            The model can be used as a template. The headers and the
            specifications need to be fulfilled.
        """
        self.shape = input_shape
        self.__autoencoder = Sequential(name="AutoEncoder")
        self.__encoder = Sequential(name="Encoder")

        self.__set_model(self.shape)

    def __set_model(self, input_shape):
        """Creation of all the layers of the network.

        The encoder is composed of three convolutional layers, with non-square
        max-pooling after each convolutional layer. The decoder is classically
        composed of three convolutional layers too, in the opposite direction.

        Args:
            input_shape (`tuple`): Input shape of the model, typically
                (_, _, 1).
        """
        # encoder layers
        self.__encoder.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
        self.__encoder.add(LeakyReLU(0.8))
        self.__encoder.add(MaxPooling2D((2, 1), padding='same'))
        self.__encoder.add(Conv2D(8, (3, 3), padding='same'))
        self.__encoder.add(LeakyReLU(0.8))
        self.__encoder.add(MaxPooling2D((2, 1), padding='same'))
        self.__encoder.add(Conv2D(1, (3, 3), padding='same'))
        self.__encoder.add(LeakyReLU(0.8))

        # decoder layers
        self.__autoencoder.add(self.__encoder)
        self.__autoencoder.add(Conv2D(16, (3, 3), padding='same'))
        self.__autoencoder.add(LeakyReLU(0.8))
        self.__autoencoder.add(UpSampling2D((2, 1)))
        self.__autoencoder.add(Conv2D(8, (3, 3), padding='same'))
        self.__autoencoder.add(LeakyReLU(0.8))
        self.__autoencoder.add(UpSampling2D((2, 1)))
        self.__autoencoder.add(Conv2D(1, (3, 3), padding='same'))

    def autoencoder(self):
        """Auto-encoder in the keras format.

        After calling this method, all the keras functions that can be applied
        on a model (compile, fit, ...) can be called on the output of this
        method.

        Returns:
            The complete auto-encoder in the keras format.
        """
        return self.__autoencoder

    def encoder(self):
        """Encoder in the keras format.

        After calling this method, all the keras functions that can be applied
        on a model (compile, fit, ...) can be called on the output of this
        method.

        Returns:
            The encoder section in the keras format.
        """
        return self.__encoder
