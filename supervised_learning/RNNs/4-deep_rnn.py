#!/usr/bin/env python3
"""4-deep_rnn.py"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Forward propagation for a deep RNN"""
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = []

    for time_step in range(t):
        x_t = X[time_step]
        h_prev_layer = h_0[:, :, :]

        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(h_prev_layer[layer], x_t)
            H[time_step + 1, layer] = h_next
            h_prev_layer[layer] = h_next
            x_t = h_next

        Y.append(y)

    return H, np.array(Y)
