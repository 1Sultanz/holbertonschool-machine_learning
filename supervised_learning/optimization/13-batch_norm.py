#!/usr/bin/env python3
"""Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """This function normalizes an unactivated 
    output of a neural network using batch normalization"""
    m = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    n_Z = (Z - m) / (np.sqrt(var) + epsilon)

    out = gamma * n_Z + beta
    return out
