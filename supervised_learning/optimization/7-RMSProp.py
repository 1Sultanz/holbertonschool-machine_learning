#!/usr/bin/env python3
"""RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """This function updates a variable using
     the RMSProp optimization algorithm"""
    update_variable = beta2 * s + (1 - beta2) * grad**2
    new_moment = var - (alpha / np.sqrt(update_variable) + epsilon) * grad
    return new_moment, update_variable
