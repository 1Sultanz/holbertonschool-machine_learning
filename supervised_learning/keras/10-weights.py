#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """This function saves a model’s weights"""
    network.save_weights(filename, save_format=save_format)

def load_weights(network, filename):
    """This function loads a model’s weights"""
    network.loads_weights(filename)
