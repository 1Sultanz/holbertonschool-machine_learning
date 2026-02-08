#!/usr/bin/env python3
"""RNN Decoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder"""

    def __init__(self, vocab, embedding, units, batch):
        """Init the class"""
        super().__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """Call the layer"""
        context_vector, attention_weights = self.attention(s_prev,
                                                           hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)
        return x, state
