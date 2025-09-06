#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """This function saves an entire model"""
    network.save(filename)

def load_model(filename):
    """This function loads an entire model"""
    return K.models.load_model(filename)
