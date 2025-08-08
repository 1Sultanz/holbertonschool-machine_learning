#!/usr/bin/env python3
"""Gettin Cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """This function concatenates two matrices along a specific axis"""
    if len(mat1[0]) == len(mat2[0]) and axis == 0:
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif len(mat1) == len(mat2) and axis == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
