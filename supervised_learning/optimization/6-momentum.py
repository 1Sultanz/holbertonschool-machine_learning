#!/usr/bin/env python3
"""Momentum Upgraded"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """This function sets up the gradient descent with
    momentum optimization algorithm in TensorFlow"""
    optimizer = tf.keras.optimizers.SGD(alpha, beta1)
    return optimizer
