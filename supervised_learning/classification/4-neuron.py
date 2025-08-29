#!/usr/bin/env python3
"""Neuron"""

import numpy as np


class Neuron:
    """This class defines a single neuron performing
    binary classifiaction"""

    def __init__(self, nx):
        """Initializes the neuron with nx inputs"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W"""
        return self.__W

    @property
    def b(self):
        """Getter for b"""
        return self.__b

    @property
    def A(self):
        """Getter for A"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        self.__A = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-self.__A))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the neuron"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""
        self.forward_prop(X)
        self.cost(Y, self.__A)
        if self.__A > 0.5:
            return 1
        else:
            return 0
