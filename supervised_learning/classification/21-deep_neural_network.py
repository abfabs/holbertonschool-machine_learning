#!/usr/bin/env python3
"""
21-deep_neural_network.py

Defines a DeepNeuralNetwork class for binary classification and implements
forward propagation, cost, evaluation, and one pass of gradient descent.
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network with private attributes."""

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of positive integers (nodes per layer).

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
            TypeError: If layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        layer = 1
        for nodes in layers:
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.__weights["W{}".format(layer)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(layer)] = np.zeros((nodes, 1))
            prev = nodes
            layer += 1

    @property
    def L(self):
        """Number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding intermediary values."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding weights and biases."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculate forward propagation.

        Args:
            X (np.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: Output of the network and the cache.
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            w_key = "W{}".format(layer)
            b_key = "b{}".format(layer)
            a_prev = self.__cache["A{}".format(layer - 1)]

            z = (
                np.matmul(self.__weights[w_key], a_prev)
                + self.__weights[b_key]
            )
            a = 1 / (1 + np.exp(-z))
            self.__cache["A{}".format(layer)] = a

        return (
            self.__cache["A{}".format(self.__L)],
            self.__cache
        )

    def cost(self, Y, A):
        """
        Calculate logistic regression cost.

        Args:
            Y (np.ndarray): Correct labels of shape (1, m).
            A (np.ndarray): Activated output of shape (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        a_safe = 1.0000001 - A
        loss = Y * np.log(A) + (1 - Y) * np.log(a_safe)
        return float((-1 / m) * np.sum(loss))

    def evaluate(self, X, Y):
        """
        Evaluate the network's predictions.

        Args:
            X (np.ndarray): Input data of shape (nx, m).
            Y (np.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: (prediction, cost)
        """
        a_last, _ = self.forward_prop(X)
        cost = self.cost(Y, a_last)
        prediction = (a_last >= 0.5).astype(int)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the network.

        Args:
            Y (np.ndarray): Correct labels of shape (1, m).
            cache (dict): Dictionary containing A0..AL activations.
            alpha (float): Learning rate.

        Updates:
            __weights (dict): Updates Wl and bl in-place.
        """
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            a_prev = cache["A{}".format(layer - 1)]
            w_key = "W{}".format(layer)
            b_key = "b{}".format(layer)

            dw = (1 / m) * np.matmul(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            w_curr = self.__weights[w_key].copy()

            self.__weights[w_key] = self.__weights[w_key] - alpha * dw
            self.__weights[b_key] = self.__weights[b_key] - alpha * db

            if layer > 1:
                a_prev_act = a_prev
                dz = np.matmul(w_curr.T, dz) * (
                    a_prev_act * (1 - a_prev_act)
                )
