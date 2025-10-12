#!/usr/bin/env python3
"""DenseNet-121"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """This function builds the DenseNet-121 architecture"""
    init = K.initializers.HeNormal(seed=0)
    input_layer = K.Input(shape=(224, 224, 3))
    nb_filters = 64

    X = K.layers.BatchNormalization()(input_layer)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(
        nb_filters,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AveragePooling2D(pool_size=7, padding='same')(X)
    X = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=init
    )(X)
    model = K.Model(inputs=input_layer, outputs=X)
    return model
