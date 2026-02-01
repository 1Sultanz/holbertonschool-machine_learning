#!/usr/bin/env python3
"""Extract Word2Vec"""
import tensorflow as tf


def gensim_to_keras(model):
    """Convert gensim word2vec to keras Embedding"""
    embedding_matrix = model.wv.vectors

    vocab_size, embedding_dim = embedding_matrix.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return layer
