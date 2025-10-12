#!/usr/bin/env python3
"""Inception Block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """This function builds an inception block as described
     in Going Deeper with Convolutions (2014)"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv_F1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(A_prev)
    conv_F3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(A_prev)
    conv_F3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(conv_F3R)
    conv_F5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(A_prev)
    conv_F5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(conv_F5R)
    max_pooling = K.layers.MaxPooling2D(
        pool_size=3,
        strides=1,
        padding="same"
    )(A_prev)
    conv_FPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(max_pooling)
    output = K.layers.Concatenate()([conv_F1, conv_F3, conv_F5, conv_FPP])
    return output

