#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """This class defines a deep neural
    network performing binary classification"""

    def __init__(self, nx, layers):
        """Class Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]

            self.weights["W" + str(i + 1)] = (
                np.random.randn(
                    layers[i], prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def cache(self):
        """Getter method for cache"""
        self.__cache

    @property
    def L(self):
        """Getter method for L"""
        self.__L

    @property
    def weights(self):
        """Getter method for weights"""
        self.__weights
