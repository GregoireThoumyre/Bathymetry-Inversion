#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################
# Train a new CNN on encoded or not TS                               #
# prediction with cnn                                                #
######################################################################

import context
import tensorflow as tf
from src.models import cnn
from src.networks.cnn import CNN

if __name__ == '__main__':
    # creation and training of the CNN, to process the data. the pre-processing
    # of the data is done using the previous encoded dataset
    name = 'Model2_test15'
    cnn1 = CNN(model=cnn.Model2((130, 480, 1), 480), batch_size=64, dataset_path='../GREGOIRE/dataset_new', encoded=True) # Adapt to your path
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    cnn1.compile()
    hcnn1 = cnn1.fit(epochs=50, repeat=1, fname=('cnn-'+name))

    # saving of the computed network metrics (only the CNN part)
    cnn1.save_losses(hcnn1, 'cnn-'+name)

