#!/usr/bin/env python3
"""Hyperparametr Tuning"""
import numpy as np


class GaussianProcess:
    """Gaussian Process Class"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Init the class"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Compute the Kernel Covariance matrix using RBF"""
        dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist)

    def predict(self, X_s):
        """Compute the mean and the variance for new datapoint
        following the GP"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu_s = np.matmul(np.matmul(K_s.T, K_inv), self.Y)
        covariance_s = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        return mu_s.reshape(1, -1), np.diagonal(covariance_s)
