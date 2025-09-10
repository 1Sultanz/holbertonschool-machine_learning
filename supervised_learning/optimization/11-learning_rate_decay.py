#!/usr/bin/env python3
"""Learning Rate Decay0"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """This function updates the learning
     rate using inverse time decay in numpy"""
    new_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return new_alpha
