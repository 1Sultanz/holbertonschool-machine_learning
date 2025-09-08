#!/usr/bin/env python3
"""Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """This function shuffles the data
     points in two matrices the same way"""
    new_X = np.random.permutation(X)
    new_Y = np.random.permutation(Y)
    return new_X, new_Y
