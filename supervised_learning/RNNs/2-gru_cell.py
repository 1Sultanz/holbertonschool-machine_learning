#!/usr/bin/env python3
"""2-gru_cell.py"""

import numpy as np


class GRUCell:
    """Recurrent Unit"""
    def __init__(self, i, h, o):
        """Initialization"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.br = np.zeros((1, h))

    def forward(self, h_prev, x_t):
        """Forward propagation"""
        conc = np.concatenate((h_prev, x_t), axis=1)

        zt = self.sigmoid(np.dot(conc, self.Wz) + self.bz)

        rt = self.sigmoid(np.dot(conc, self.Wr) + self.br)

        conc_reset = np.concatenate((rt * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(conc_reset, self.Wh) + self.bh)

        h_next = (1 - zt) * h_prev + zt * h_tilde

        y = np.dot(h_next, self.Wy) + self.by
        softmax_y = self.softmax(y)

        return h_next, softmax_y

    @staticmethod
    def sigmoid(x):
        """Sigmoid"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
