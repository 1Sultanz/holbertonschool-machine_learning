#!/usr/bin/env python3
"""1-rnn.py"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """forward propagation"""
    t, m, i = X.shape
    h_prev = h_0
    H = [h_0]
    Y = []
    for k in range(t):
        x_t = X[k]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H.append(h_next)
        Y.append(y)
        h_prev = h_next
    return np.array(H), np.array(Y)
