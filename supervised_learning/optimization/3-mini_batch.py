#!/usr/bin/env python3
"""Mini Batch"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """This function creates mini-batches to be used for
    training a neural network using mini-batch gradient descent"""
    X_shuffle, Y_shuffle = shuffle_data(X, Y)
    mini_batch = []
    i = len(X)

    for first_index in range(0, i, batch_size):
        last_index = min(first_index + batch_size, i)
        X_batch = X_shuffle[first_index:last_index]
        Y_batch = Y_shuffle[first_index:last_index]
        mini_batch.append((X_batch, Y_batch))
    return mini_batch
