#!/usr/bin/env python3
"""RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """This function updates a variable using
     the RMSProp optimization algorithm"""
    s = beta2 * s + (1 - beta2) * grad**2
    var = var - (alpha / np.sqrt(s + epsilon)) * grad
    return var, s
