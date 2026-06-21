#!/usr/bin/env python3
"""Policy gradient module for reinforcement learning."""
import numpy as np


def policy(matrix, weight):
    """Compute the policy with a weight of a matrix using softmax.

    Args:
        matrix: current observation/state matrix
        weight: matrix of random weights

    Returns:
        Softmax probability distribution over actions
    """
    z = matrix @ weight
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Compute the Monte-Carlo policy gradient.

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Returns:
        action: the chosen action
        gradient: the policy gradient
    """
    state = state[np.newaxis, :]
    probs = policy(state, weight)

    action = np.random.choice(probs.shape[1], p=probs[0])

    d_softmax = probs.copy()
    d_softmax[0, action] -= 1
    gradient = -state.T @ d_softmax

    return action, gradient
