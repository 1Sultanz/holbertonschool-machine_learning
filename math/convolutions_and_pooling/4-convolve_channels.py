#!/usr/bin/env python3
"""Convolution with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """This function performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    image_pad = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0
    )
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(kernel * region, axis=(1, 2, 3))
    return output
