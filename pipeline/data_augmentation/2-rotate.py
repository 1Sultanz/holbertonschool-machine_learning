#!/usr/bin/env python3
"""2-rotate.py"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    image: 3D tf.Tensor (height, width, channels)

    Returns: rotated image (tf.Tensor)
    """
    return tf.image.rot90(image)
