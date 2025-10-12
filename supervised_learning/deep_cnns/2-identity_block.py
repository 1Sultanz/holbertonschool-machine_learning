#!/usr/bin/env python3
"""Identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """This function builds an identity block as described in
     Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    conv = K.layers.Conv2D(
        filters=F11, kernel_size=1, padding="same",
        kernel_initializer=init
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
    batch_normalization3 = K.layers.BatchNormalization()(conv3)
    add = K.layers.Add()([batch_normalization3, A_prev])
    activation3 = K.layers.Activation("relu")(add)
    return activation3
