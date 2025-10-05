#!/usr/bin/env python3
"""Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """This function  performs back propagation
     over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((A_prev.shape))
    for e in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    if mode == 'avg':
                        average_dA = dA[e, i, j, k] / (kh * kw)
                        dA_prev[e, i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        k] += average_dA
                    elif mode == 'max':
                        slice = A_prev[e, i * sh:i * sh + kh,
                        j * sw:j * sw + kw, k]
                        mask = (slice == np.max(slice))
                        dA_prev[e, i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        k] += mask * dA[e, i, j, k]
    return dA_prev
