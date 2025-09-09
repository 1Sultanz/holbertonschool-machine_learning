#!/usr/bin/env python3
"""Adam Optimizer"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """This function updates a variable in place
     using the Adam optimization algorithm"""
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * grad

    v_hat = v_new / (1 - beta1 ** t)
    s_hat = s_new / (1- beta2 ** t)

    var_new = var - alpha * (v_hat / (np.sqrt(s_hat) + epsilon))
    return var_new, v_hat, s_hat
