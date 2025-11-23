#!/usr/bin/env python3
"""Multivariate Probability"""
import numpy as np


class MultiNormal:
    """Multivariate Normal Distribution"""
    def __init__(self, data):
        """Class Constructor"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = (data_centered @ data_centered.T) / (n - 1)
        self.d = d

    def pdf(self, x):
        """Probability Density Function"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")
        diff = x - self.mean
        det_cov = np.linalg.det(self.cov)
        if det_cov == 0:
            return 0
        inv_cov = np.linalg.inv(self.cov)
        exponent = -0.5 * (diff.T @ inv_cov @ diff)
        norm_const = 1 / np.sqrt((2 * np.pi) ** self.d * det_cov)
        pdf_value = float(norm_const * np.exp(exponent))
        return pdf_value
