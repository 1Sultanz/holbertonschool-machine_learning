#!/usr/bin/env python3
"""Ridin Bareback"""


def mat_mul(mat1, mat2):
    """This function performs matrix multiplication"""
    if len(mat1[0]) == len(mat2):
        mul = [sum(mat1[i][k] * mat2[k][j] for k in range(len(mat1[0])))
                for i in range(len(mat1))
                for j in range(len(mat2[0]))]
        return mul
    else:
        None
