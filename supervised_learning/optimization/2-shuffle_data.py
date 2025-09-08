#!/usr/bin/env python3
"""Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """This function shuffles the data
     points in two matrices the same way"""
    if len(X) != len(Y):
        return None
    else:
        p = np.random.permutation(len(X))
    return X[p], Y[p]
