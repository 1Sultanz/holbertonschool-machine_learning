#!/usr/bin/env python3
"""This module includes the function
that has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        max_steps is the maximum number of steps in the episode
        You need to update 0-load_env.py to add render_mode="ansi"
        Each state of the board should be displayed via the console
        You should always exploit the Q-table
        Ensure that the final state of the environment
         is also displayed after the episode concludes.
    Returns:
        total_rewards: the total rewards for the episode
        rendered_outputs: list of rendered board states per step
    """
    state, _ = env.reset()
    total_reward = 0.0
    rendered_outputs = []
    rendered_outputs.append(env.render())

    for steps in range(max_steps):
        # We have already explored whole environment
        # that's why we are going to exploit only
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rendered_outputs.append(env.render())
        total_reward += reward
        state = new_state
        if done:
            break

    return total_reward, rendered_outputs
