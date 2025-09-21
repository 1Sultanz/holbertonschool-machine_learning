#!/usr/bin/env python3
"""Gradient Descent with L2 Regulsrization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """This function updates the weights and biases of a neural
     network using gradient descent with L2 regularization"""
    dZ = cache.get("A{}".format(L)) - Y
    m = Y.shape[1]

    for i in reversed(range(1, L + 1)):
        W = weights.get("W{}".format(i))
        b = weights.get("b{}".format(i))
        A_prev = cache.get("A{}".format(i - 1))

        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dW = (1/m) * (dZ @ A_prev.T) + (lambtha/m) * W
        dZ = (W.T @ dZ) * (1 - A_prev ** 2)

        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db
