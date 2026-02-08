#!/usr/bin/env python3
"""RNN Encoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder"""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the class"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def initialize_hidden_state(self):
        """Initialize the first hidden state (tensor)"""
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """Call the layer"""
        x_embedded = self.embedding(x)
        output, hidden = self.gru(x_embedded, initial_state=initial)
        return output, hidden
