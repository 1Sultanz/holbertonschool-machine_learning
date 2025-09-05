#!/usr/bin/env python3
"""Neuron"""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """This class defines a single neuron performing
    binary classification"""

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
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)  # vectorized threshold
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        self.__W = self.__W - alpha * np.matmul(
            (A - Y), X.T) / X.shape[1]
        self.__b = self.__b - alpha * np.sum(A - Y) / X.shape[1]
        return self.__W, self.__b


    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Train the neuron."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")


        costs, iteration_list = [], []

        for iteration in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if (iteration % step == 0) or (iteration == iterations):
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {iteration} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iteration_list.append(iteration)

        if graph:
            plt.plot(iteration_list, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()

        self.forward_prop(X)
        return self.evaluate(X, Y)
