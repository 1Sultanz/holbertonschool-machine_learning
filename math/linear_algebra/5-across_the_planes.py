#!/usr/bin/env python3
"""Across The Planes"""


def add_matrices2D(mat1, mat2):
    """This function adds two matrices element-wise"""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        result = []
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            result.append(row)
        return result
    else:
        None
