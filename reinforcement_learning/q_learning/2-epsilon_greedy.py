#!/usr/bin/env python3
"""This module includes the function that
uses epsilon-greedy to determine the next action"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Args:
        Q is a numpy.ndarray containing the q-table
        state is the current state
        epsilon is the epsilon to use for the calculation
        You should sample p with numpy.random.uniform
         to determine if your algorithm should explore or exploit
        If exploring, you should pick the next action with
         numpy.random.randint from all possible actions
    Returns:
        the next action index
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[state])
