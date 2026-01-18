#!/usr/bin/env python3
"""0-rnn_cell.py"""


import numpy as np


class RNNCell:
    """Cellule de RNN"""
    def __init__(self, i, h, o):
        """initialisation"""
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """propagation en avant"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(conc, self.Wh)+self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        max_y = np.max(y, axis=1, keepdims=True)
        exp_y = np.exp(y - max_y)
        sum_y = np.sum(exp_y, axis=1, keepdims=True)
        softmax_y = exp_y / sum_y

        return h_next, softmax_y
