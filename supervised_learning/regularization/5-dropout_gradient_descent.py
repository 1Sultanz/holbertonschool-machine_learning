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

        if i > 1:
            dA_prev = W.T @ dZ
            D = cache['D' + str(i - 1)]
            dA_prev = dA_prev @ D
            dA_prev /= keep_prob

            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
