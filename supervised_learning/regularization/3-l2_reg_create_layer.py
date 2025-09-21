#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """This function creates a neural network layer in
     tensorFlow that includes L2 regularization"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                         distribution='normal')
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
