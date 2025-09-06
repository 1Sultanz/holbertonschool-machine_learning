#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """This function tests a neural network."""
    return network.evaluate(data, labels, verbose=verbose)
