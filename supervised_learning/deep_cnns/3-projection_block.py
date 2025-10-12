#!/usr/bin/env python3
"""Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """This function builds a projection block as described in Deep
     Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    conv = K.layers.Conv2D(
        filters=F11, kernel_size=1, padding="same",
        strides=s, kernel_initializer=init
    )(A_prev)
    batch_normalization = K.layers.BatchNormalization()(conv)
    activation = K.layers.Activation("relu")(batch_normalization)

    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=3, padding="same",
        kernel_initializer=init
    )(activation)
    batch_normalization2 = K.layers.BatchNormalization()(conv2)
    activation2 = K.layers.Activation("relu")(batch_normalization2)

    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=1, padding="same",
        kernel_initializer=init
    )(activation2)
    conv4 = K.layers.Conv2D(
        filters=F12, kernel_size=1, padding="same",
        strides=s, kernel_initializer=init
    )(A_prev)
    batch_normalization3 = K.layers.BatchNormalization()(conv3)
    batch_normalization4 = K.layers.BatchNormalization()(conv4)
    add = K.layers.Add()([batch_normalization3, batch_normalization4])
    activation3 = K.layers.Activation("relu")(add)
    return activation3
