#!/usr/bin/env python3
"""Transition Layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """This function builds a transition layer"""
    init = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)
    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=init
    )(activation)
    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=2,
        padding='same'
    )(conv)
    return average_pooling2d, nb_filters
