#!/usr/bin/env python3
"""0-flip.py"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    image: 3D tf.Tensor (height, width, channels)

    Returns: flipped image (tf.Tensor)
    """
    return tf.image.flip_left_right(image)
