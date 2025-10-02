#!/usr/bin/env python3
"""Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """This function performs back propagation
    over a convolutional layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
        A_prev = np.pad(
            A_prev,
            pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode="constant",
            constant_values=0
        )
    elif padding == "valid":
        ph = 0
        pw = 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    dA_prev = np.zeros((A_prev.shape))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for e in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    dA_prev[e, i * sh:i * sh + kh, j * sw:j * sw + kw, :] \
                        += (W[:, :, :, k] * dZ[e, i, j, k])
                    dW[:, :, :, k] += (A_prev[e, i * sh:i * sh + kh,
                                              j * sw:j * sw + kw, :] *
                                       dZ[e, i, j, k])
    if padding == "same":
        dA = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA = dA_prev
    return dA, dW, db
