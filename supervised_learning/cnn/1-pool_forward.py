#!/usr/bin/env python3
"""Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """This function performs forward propagation
     over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape.shape
    sh, sw = stride
    out_h = (h_prev - kh) // sh + 1
    out_w = (w_prev - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, c_prev))
    for i in range(out_h):
        for j in range(out_w):
            if mode == "max":
                output[:, i, j, :] = np.max(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
            elif mode == "avg":
                output[:, i, j, :] = np.mean(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
    return output
