#!/usr/bin/env python3
"""Adam Upgraded"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """This function sets up the Adam
     optimization algorithm in TensorFlow"""
    optimizer = tf.keras.optimizers.Adam(
        alpha, beta1, beta2, epsilon
    )
    return optimizer
