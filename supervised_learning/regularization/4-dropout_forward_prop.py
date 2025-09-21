#!/usr/bin/env pyton3
"""Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """This function conducts forward
     propagation using Dropout"""
    cache = {"A0": X}
    for i in range(1, L+1):
        W = weights.get("W{}".format(i))
        b = weights.get("b{}".format(i))
        A_prev = cache.get("A{}".format(i-1))
        Z = W @ A_prev + b
        if i == L+1:
            exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exps / np.sum(exps, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A = (A*D)/keep_prob
            cache["D{}".format(i)] = D
        cache["A{}".format(i)] = A
    return cache
