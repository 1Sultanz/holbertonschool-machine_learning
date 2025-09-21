#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """This function updates the weights of a neural network
     with Dropout regularization using gradient descent"""
    dZ = cache.get("A{}".format(L)) - Y
    m = Y.shape[1]

    for i in reversed(range(1, L + 1)):
        W = weights.get("W{}".format(i))
        b = weights.get("b{}".format(i))
        A_prev = cache.get("A{}".format(i-1))

        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db

        if i > 1:
            dZ = W.T @ dZ
            A_prev_current = cache.get("A{}".format(i-1))
            dZ = dZ * (1 - A_prev_current ** 2)
            D = cache.get("D{}".format(i-1))
            dZ = dZ * D / keep_prob
