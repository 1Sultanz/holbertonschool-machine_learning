#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """This function creates a neural network layer in
     tensorFlow that includes L2 regularization"""
    def l2(weights):
        return lambtha * tf.reduce_sum(tf.square(weights))

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2
    )
    return layer(prev)
