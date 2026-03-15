#!/usr/bin/env python3
"""
Train a DQN agent to play Atari Breakout using keras-rl2
and gymnasium. Saves the final policy network as policy.h5
"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Permute
)
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class CompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to make Gymnasium compatible with keras-rl2
    by updating reset, step, and render signatures
    """

    def reset(self, **kwargs):
        """Reset the environment and return only the state"""
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """
        Step the environment and return
        (obs, reward, done, info) for keras-rl2 compatibility
        """
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, **kwargs):
        """Render the environment"""
        return self.env.render(**kwargs)


def build_model(input_shape, nb_actions):
    """
    Build the CNN model for DQN
    Args:
        input_shape: shape of the input (frames, height, width)
        nb_actions: number of possible actions
    Returns:
        compiled Keras model
    """
    model = Sequential([
        Permute((2, 3, 1), input_shape=input_shape),
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])
    return model


def create_env():
    """
    Create and wrap the Breakout environment
    Returns:
        wrapped environment compatible with keras-rl2
    """
    env = gym.make(
        'ALE/Breakout-v5',
        render_mode=None
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True
    )
    env = FrameStackObservation(env, stack_size=4)
    env = CompatibilityWrapper(env)
    return env


def main():
    """Main training function"""
    env = create_env()

    nb_actions = env.action_space.n
    input_shape = (4, 84, 84)

    model = build_model(input_shape, nb_actions)
    model.summary()

    memory = SequentialMemory(
        limit=1000000,
        window_length=1
    )

    policy = EpsGreedyQPolicy(
        eps=1.0
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy,
        gamma=0.99,
        batch_size=32,
        train_interval=4,
        delta_clip=1.0
    )

    dqn.compile(
        optimizer=Adam(learning_rate=0.00025),
        metrics=['mae']
    )

    dqn.fit(
        env,
        nb_steps=10000000,
        visualize=False,
        verbose=2,
        log_interval=10000
    )

    dqn.save_weights('policy.h5', overwrite=True)
    print("Training complete. Policy saved to policy.h5")

    env.close()


if __name__ == '__main__':
    main()
