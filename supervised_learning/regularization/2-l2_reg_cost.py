#!/usr/bin/env python3
"""L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """This function calculates the cost of
     a neural network with L2 regularization"""
    reg_losses = tf.reduce_sum(model.losses)
    total_cost = cost + reg_losses
    return total_cost
