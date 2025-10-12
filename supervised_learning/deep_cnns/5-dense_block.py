#!/usr/bin/env python3
"""Dense Block"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """This function builds a dense block"""
    init = K.initializers.he_normal(seed=0)
    i = 0
    while i < layers:
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=init,
                                )(layer)
        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=init,
                                )(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate
        i += 1
    return X, nb_filters
