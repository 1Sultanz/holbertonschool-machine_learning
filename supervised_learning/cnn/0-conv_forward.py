#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """This function performs forward propagation over
     a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(np.ceil((h_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph = 0
        pw = 0
    elif isinstance(padding, tuple):
        ph, pw = padding
    image_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0
    )
    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1
    A = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            region = image_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            for k in range(c_new):
                A[:, i, j, k] = np.sum(
                    region * W[:, :, :, k], axis=(1, 2, 3)
                )
    Z = A + b
    return activation(Z)
