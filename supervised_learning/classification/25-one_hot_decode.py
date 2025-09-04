#!/usr/bin/env python3
"""One-Hot Encode"""

import numpy as np


def one_hot_decode(one_hot):
    """This function converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray or one_hot.ndim != 2:
        return None
    return one_hot.transpose().argmax(axis=1)
