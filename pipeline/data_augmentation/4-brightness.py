#!/usr/bin/env python3
"""4-brightness.py"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    image: 3D tf.Tensor (height, width, channels)
    max_delta: float, maximum brightness change

    Returns: brightness-adjusted image (tf.Tensor)
    """
    return tf.image.random_brightness(image, max_delta)
