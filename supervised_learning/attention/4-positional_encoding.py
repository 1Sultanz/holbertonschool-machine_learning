#!/usr/bin/env python3
"""Positional Encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Compute the positional encoding"""
    P = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in np.arange(int(dm / 2)):
            denominator = np.power(10000, 2 * i / dm)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P
