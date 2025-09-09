#!/usr/bin/env python3
"""Momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """This function updates a variable using the
     gradient descent with momentum optimization algorithm"""
    update_variable = beta1 * v + (1 - beta1) * grad
    new_moment = var - alpha * update_variable
    return new_moment, update_variable
