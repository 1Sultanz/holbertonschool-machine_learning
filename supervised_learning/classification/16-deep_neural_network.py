#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """This class defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a postitive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if l <= 0:
                raise TypeError("layers must be a list of positive integers")

            if l == 1:
                prev = nx
            else:
                prev = layers[l - 2]

            self.weights["W" + str(l)] = {
                np.randm.randn(layers[i-1], prev) * np.sqrt(2 / prev)
            }

            self.weights["b" + str(l)] = np.zeros((layers[l-1], 1))
