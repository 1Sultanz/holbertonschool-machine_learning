#!/usr/bin/env python3
"""8-bi_rnn.py"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward and backward RNN"""
    t, m, i = X.shape
    m, h = h_0.shape

    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    h_prev = h_0
    for i in range(t):
        x_t = X[i]
        h_prev = bi_cell.forward(h_prev, x_t)
        Hf[i] = h_prev

    h_next = h_t
    for i in reversed(range(t)):
        x_t = X[i]
        h_next = bi_cell.backward(h_next, x_t)
        Hb[i] = h_next

    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)

    return H, Y
