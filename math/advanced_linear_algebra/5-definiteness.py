#!/usr/bin/env python3
"""Definiteness"""
import numpy as np


def definiteness(matrix):
    """Definiteness"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0:
        return None
    row, col = matrix.shape
    if not row == col:
        return None
    if not np.all(matrix == matrix.T):
        return None

    eigen_values, _ = np.linalg.eig(matrix)

    min_value = np.min(eigen_values)
    max_value = np.max(eigen_values)

    if max_value > 0 and min_value > 0:
        return "Positive definite"
    elif max_value > 0 and min_value == 0:
        return "Positive semi-definite"
    elif max_value < 0 and min_value < 0:
        return "Negative definite"
    elif max_value == 0 and min_value < 0:
        return "Negative semi-definite"
    elif max_value > 0 > min_value:
        return "Indefinite"
    return None
