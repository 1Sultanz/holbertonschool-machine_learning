#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """This Function performs a same
    convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph_top = kh // 2
    ph_bottom = kh - ph_top - 1
    pw_left = kw // 2
    pw_right = kw - pw_left - 1
    image_pad = np.pad(
        images,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_right, pw_left)),
        mode="constant",
        constant_values=0
    )
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            region = image_pad[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
