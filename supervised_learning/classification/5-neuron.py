#!/usr/bin/env python3
"""
Neuron class with forward propagation, cost, evaluation, and gradient descent
"""

import numpy as np


class Neuron:
    """Single neuron performing binary classification"""

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Performs forward propagation with sigmoid activation"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        Returns prediction (1/0) and cost
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent and updates __W and __b
        X: input data (nx, m)
        Y: true labels (1, m)
        A: activated outputs (1, m)
        alpha: learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
