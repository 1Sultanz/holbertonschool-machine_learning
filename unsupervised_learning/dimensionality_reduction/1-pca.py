#!/usr/bin/env python3
"""Dimensinality Reduction"""
import numpy as np


def pca(X, ndim):
    """Compute the PCA"""
    X = X - np.mean(X, axis=0)
    _, __, Vt = np.linalg.svd(X)
    return np.matmul(X, Vt.T[..., :ndim])
