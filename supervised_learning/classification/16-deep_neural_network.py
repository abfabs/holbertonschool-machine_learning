#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        # Public attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # He initialization (one loop)
        prev = nx
        for l, nodes in enumerate(layers, start=1):
            self.weights[f"W{l}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            self.weights[f"b{l}"] = np.zeros((nodes, 1))
            prev = nodes
