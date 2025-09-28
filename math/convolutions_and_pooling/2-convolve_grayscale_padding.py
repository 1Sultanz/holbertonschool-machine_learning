#!/usr/bin/env python3
"""Convolution with padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """This function performs a convolution
     on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    image_pad = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode="constant",
        constant_values=0
    )
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image_pad[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
