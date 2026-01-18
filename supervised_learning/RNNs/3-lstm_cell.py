#!/usr/bin/env python3
"""3-lstm_cell.py"""

import numpy as np


class LSTMCell:
    """LSTMCell"""
    def __init__(self, i, h, o):
        """initialization"""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """forward propagation"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        ft = self.sigmoid(np.dot(conc, self.Wf) + self.bf)
        ut = self.sigmoid(np.dot(conc, self.Wu) + self.bu)
        C_tilde = np.tanh(np.dot(conc, self.Wc) + self.bc)
        Ct = np.multiply(ft, c_prev) + np.multiply(ut, C_tilde)
        ot = self.sigmoid(np.dot(conc, self.Wo) + self.bo)
        ht = np.multiply(ot, np.tanh(Ct))
        y = np.dot(ht, self.Wy) + self.by
        y = self.softmax(y)

        return ht, Ct, y

    @staticmethod
    def sigmoid(x):
        """Sigmoid"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
