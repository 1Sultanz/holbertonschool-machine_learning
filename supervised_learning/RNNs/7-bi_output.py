#!/usr/bin/env python3
"""7-bi_output.py"""

import numpy as np


class BidirectionalCell:
    """classe de BidirectionalCell"""
    def __init__(self, i, h, o):
        """initialisation"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2*h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation"""
        h_next = np.tanh(np.dot(np.concatenate((
            h_prev, x_t), axis=1), self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """bacward"""
        h_prev = h_next = np.tanh(np.dot(np.concatenate((
            h_next, x_t), axis=1), self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """Output"""
        Y = self.softmax(np.dot(H, self.Wy) + self.by)
        return Y

    @staticmethod
    def softmax(x):
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=2, keepdims=True))
        return exp_x / np.sum(exp_x, axis=2, keepdims=True)
