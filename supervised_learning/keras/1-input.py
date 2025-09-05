#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """This function builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    drop = 1 - keep_prob
    n = len(layers)
    x = K.layers.Dense(
        layers[0], activation=activations[0],
        kernel_regularizer=L2
    )(inputs)

    if n > 1:
        x = K.layers.Dropout(drop)(x)

    for i in range(1, n):
        x = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=L2
        )(x)

        if i < n-1:
            x = K.layers.Dropout(drop)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
