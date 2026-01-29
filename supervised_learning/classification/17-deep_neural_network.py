#!/usr/bin/env python3
"""
17-deep_neural_network.py

Defines a DeepNeuralNetwork class (privatized) for binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification with private attrs."""

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
        """Get the number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Get the cache dictionary of intermediary values."""
        return self.__cache

    @property
    def weights(self):
        """Get the weights and biases dictionary."""
        return self.__weights
