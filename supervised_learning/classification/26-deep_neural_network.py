#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np
import pickle


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

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if i == 0 else layers[i - 1]

            self.__weights["W" + str(i + 1)] = (
                np.random.randn(
                    layers[i], prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def cache(self):
        """Getter method for cache"""
        return self.__cache

    @property
    def L(self):
        """Getter method for L"""
        return self.__L

    @property
    def weights(self):
        """Getter method for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Forward Propagation"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            A_prev = self.__cache["A" + str(i - 1)]
            b = self.__weights["b" + str(i)]
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(
                -(np.dot(W, A_prev) + b)))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """The cost of the model"""
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluate"""
        self.forward_prop(X)
        A = self.__cache['A{}'.format(self.__L)]
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent"""
        m = len(Y[0])
        AL = cache['A{}'.format(self.__L)]
        dZl = AL - Y
        for i in range(self.__L, 0, -1):
            Al = cache['A{}'.format(i-1)]
            dwl = (dZl @ Al.T) / m
            dbl = (np.sum(dZl, axis=1, keepdims=True)) / m

            Al_prev = cache['A{}'.format(i-1)]
            Wl = self.__weights['W{}'.format(i)]
            if i > 1:
                dZl = (Wl.T @ dZl) * (Al_prev * (1-Al_prev))
            self.__weights['W{}'.format(i)] -= alpha * dwl
            self.__weights['b{}'.format(i)] -= alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Training for Deep Neural Network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
        return self.evaluate(X, Y)\

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename
