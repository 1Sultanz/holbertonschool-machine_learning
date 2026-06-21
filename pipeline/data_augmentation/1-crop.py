#!/usr/bin/env python3
"""1-crop.py"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    image: 3D tf.Tensor (height, width, channels)
    size: tuple (new_height, new_width, channels)

    Returns: cropped image (tf.Tensor)
    """
    return tf.image.random_crop(image, size=size)
