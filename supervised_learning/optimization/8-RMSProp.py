#!/usr/bin/env python3
"""RMSProp Upgraded"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """This function sets up the RMSProp
     optimization algorithm in TensorFlow"""
    optimizer = tf.keras.optimizers.RMSprop(
        alpha, beta2, epsilon
    )
    return optimizer
