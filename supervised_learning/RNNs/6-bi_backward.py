#!/usr/bin/env python3
"""5-bi_forward.py"""

import numpy as np


class BidirectionalCell:
    """BidirectionalCell"""
    def __init__(self, i, h, o):
        """initialization"""
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
