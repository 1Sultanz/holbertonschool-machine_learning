#!/usr/bin/env python3
"""3-contrast.py"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    image: 3D tf.Tensor (height, width, channels)
    lower: float, lower bound for contrast factor
    upper: float, upper bound for contrast factor

    Returns: contrast-adjusted image (tf.Tensor)
    """
    return tf.image.random_contrast(image, lower, upper)
