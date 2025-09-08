#!/usr/bin/env python3
"""Normalization Constants"""
import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """This function calculates the normalization
    constants of a matrix"""
    mean, variance = tf.nn.moments(X, axes=[0])
    std = tf.sqrt(variance)
    return mean, std
