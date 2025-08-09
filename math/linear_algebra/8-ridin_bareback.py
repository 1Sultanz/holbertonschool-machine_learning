#!/usr/bin/env python3
"""Ridin Barebac"""


def mat_mul(mat1, mat2):
    """This function performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    return [
        [sum(mat1[i][k] * mat2[k][j] for k in range(len(mat1[0])))
         for j in range(len(mat2[0]))]
        for i in range(len(mat1))
    ]
