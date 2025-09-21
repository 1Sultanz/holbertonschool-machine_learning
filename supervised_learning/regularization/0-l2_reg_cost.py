#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """This function calculates the cost of
     a neural network with L2 regularization"""
    l2 = 0

    for i in range(1, L + 1):
        l2 += np.sum(weights.get("W{}".format(i)) ** 2)
    return cost + (lambtha/(2*m)) * l2
