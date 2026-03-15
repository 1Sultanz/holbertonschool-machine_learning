#!/usr/bin/env python3
"""This module includes the function that initializes the Q-table"""
import numpy as np


def q_init(env):
    """
    Args:
        env is the FrozenLakeEnv instance
    Returns:
        the Q-table as a numpy.ndarray of zeros
    """
    states = env.observation_space.n
    actions = env.action_space.n
    q_table = np.zeros((states, actions))
    return q_table
