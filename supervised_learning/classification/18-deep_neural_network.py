#!/usr/bin/env python3
"""
18-deep_neural_network.py

Defines a DeepNeuralNetwork class for binary classification and implements
forward propagation.
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
            tuple: (A_L, cache) where A_L is the activated output of the last
            layer and cache is the dictionary of all activations.
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            w_key = "W{}".format(layer)
            b_key = "b{}".format(layer)
            a_prev = self.__cache["A{}".format(layer - 1)]

            z = np.matmul(self.__weights[w_key], a_prev) + self.__weights[b_key]
            a = 1 / (1 + np.exp(-z))

            self.__cache["A{}".format(layer)] = a

        return self.__cache["A{}".format(self.__L)], self.__cache
