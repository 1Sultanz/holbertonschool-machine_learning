#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """This function  makes a prediction using a neural network"""
    if verbose:
        prediction = network.predict(data, verbose=1)
    else:
        prediction = network.predict(data, verbose=0)
    return prediction
