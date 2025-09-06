#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """This function converts a label vector into a one-hot matrix"""
    return K.utils.to_categorical(labels, classes)
