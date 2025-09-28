#!/usr/bin/env python3
"""Strided convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """This function performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == "same":
        ph = np.ceil(((h - 1) * sh + kh - h) / 2)
        pw = np.ceil(((w - 1) * sw + kw - w) / 2)
    elif padding == "valid":
        ph = 0
        pw = 0
    elif padding is tuple:
        ph, pw = padding

    image_pad = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode="constant",
        constant_values=0
    )
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(kernel * region, axis=(1, 2))
    return output
