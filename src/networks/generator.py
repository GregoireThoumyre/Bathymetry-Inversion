#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: generator.py |
Created on the 2019-02-22 |
Github: https://github.com/pl19n72019

This file contains generator classes, they allow us to load the data batch per
batch.
"""

import glob
import numpy as np

from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize

class GeneratorAutoencoder(Sequence):
    """Manages the batched data during the fitting. This class is adapted to
    the autoencoder network and load the data batch per batch to avoid
    memory filling. It is absolutely necessary to work with a generator if
    you want to fit the autoencoder on a big size data.
    """

    def __init__(self, ts_names, batch_size):
        """Initialize the object with the names of all the timstack
        files to process. It will rely on them to load their contents
        batch per batch.

        Args:
            ts_names (list(str)): The list of the timestack files.
            batch_size (int): Size of the batch.
        """
        self.timestacks = ts_names
        self.batch_size = batch_size

    def __len__(self):
        """Gives the number of steps per epoch. This is an inherited function
        from Sequence.

        Returns:
            Returns the number (int) of steps per epoch.
        """
        return int(np.ceil(len(self.timestacks) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Uses the names of the timestack files to load `batch_size` ones.
        This function is inherited from Sequence. All the pre-processing (croping, 
        reshaping, etc) needed is done here.

        Args:
            idx (int): The number of the batch to load in memory.

        Returns:
            Returns a tuple of identical timestack lists.
        """
        batch_ts = self.timestacks[idx * self.batch_size:(idx + 1) * self.batch_size]

        ts = []
        for data in batch_ts:
            ts.append(np.load(data)) #[200:])
        (width, height) = ts[0].shape
        ts = np.array(ts).reshape((len(ts),width,height,1))
        return (ts, ts)


class GeneratorCNN(Sequence):
    """Manages the batched data during the fitting. This class is adapted to
    the cnn network and load the data batch per batch to avoid memory filling.
    It is absolutely necessary to work with a generator if
    you want to fit the CNN on a big size data.
    """

    def __init__(self, ts_names, b_names, batch_size):
        """Initialize the object with the names of all the timstack and 
        bathymetry files to process. It will rely on them to load their
        contents batch per batch.

        Args:
            ts_names (list(str)): The list of the timestack files.
            b_names (list(str)): The list of the bathymetry files.
            batch_size (int): Size of the batch.
        """
        self.timestacks = ts_names
        self.bathy = b_names
        self.batch_size = batch_size

    def __len__(self):
        """Gives the number of steps per epoch. This is an inherited function
        from Sequence.

        Returns:
            Returns the number (int) of steps per epoch.
        """
        return int(np.ceil(len(self.timestacks) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Uses the names of the timestack and bathymetry files to load `batch_size` ones.
        This function is inherited from Sequence. All the pre-processing (croping, 
        reshaping, etc) needed is done here.

        Args:
            idx (int): The number of the batch to load in memory.

        Returns:
            Returns a tuple composed of a timestack list and associated
                bathymetry one.
        """
        batch_ts = self.timestacks[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_b = self.bathy[idx * self.batch_size:(idx + 1) * self.batch_size]

        ts = []
        b = []
        for data_ts, data_gt in zip(batch_ts, batch_b):
            ts.append(np.load(data_ts))
            b.append(np.load(data_gt))
        (width, height) = ts[0].shape
        ts = np.array(ts).reshape((len(ts),width,height,1))
        b = np.array(b)
        return (ts, b)
