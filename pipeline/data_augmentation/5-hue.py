#!/usr/bin/env python3
"""5-hue.py"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    image: 3D tf.Tensor (height, width, channels)
    delta: float, amount to shift hue (-1.0 to 1.0 typically)

    Returns: hue-adjusted image (tf.Tensor)
    """
    return tf.image.adjust_hue(image, delta)
