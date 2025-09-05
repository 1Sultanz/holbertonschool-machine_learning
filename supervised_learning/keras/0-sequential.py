#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""

import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """This function builds a neural network with the Keras library"""
    model = k.Sequential()
    L2 = k.regularizers.l2(lambtha)
    n = len(layers)
    drop = 1 - keep_prob

    for i in range(n):
        if i == 0:
            model.add(k.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=L2, input_dim=nx
            ))
        else:
            model.add(k.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=L2
            ))

        if i < n-1:
            model.add(k.layers.Dropout(drop))
    return model
