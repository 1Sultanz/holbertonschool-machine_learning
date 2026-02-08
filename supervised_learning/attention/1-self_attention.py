#!/usr/bin/env python3
"""Self Attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self Attention for Encoder Decoder architechture"""

    def __init__(self, units):
        """Initialize the class"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Call the layer"""
        s_prev = tf.expand_dims(s_prev, 1)

        W = self.W(s_prev)
        U = self.U(hidden_states)

        alignement_output = self.V(tf.math.tanh(W + U))
        weights = tf.math.softmax(alignement_output, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
