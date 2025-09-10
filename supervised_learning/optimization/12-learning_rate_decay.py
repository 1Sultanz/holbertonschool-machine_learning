#!/usr/bin/env python3
"""Learning Rate Decay Upgraded"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """This function creates a learning rate decay
     operation in tensorflow using inverse time decay"""
    optimizer = tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha, decay_step, decay_rate
    )
    return optimizer
