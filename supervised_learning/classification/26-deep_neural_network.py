#!/usr/bin/env python3
"""
26-deep_neural_network.py

Deep neural network with persistence support (save/load).
"""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Deep neural network with private attributes."""

    def __init__(self, nx, layers):
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
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation for the network.

        Args:
            X (np.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (A_L, cache)
                A_L is the activated output of the last layer
                cache is the dictionary of intermediary activations
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            w = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            a_prev = self.__cache["A{}".format(layer - 1)]

            z = np.matmul(w, a_prev) + b
            self.__cache["A{}".format(layer)] = 1 / (1 + np.exp(-z))

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        a_safe = 1.0000001 - A
        loss = Y * np.log(A) + (1 - Y) * np.log(a_safe)
        return float((-1 / m) * np.sum(loss))

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            a_prev = cache["A{}".format(layer - 1)]
            w_key = "W{}".format(layer)
            b_key = "b{}".format(layer)

            dw = (1 / m) * np.matmul(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            w_copy = self.__weights[w_key].copy()

            self.__weights[w_key] -= alpha * dw
            self.__weights[b_key] -= alpha * db

            if layer > 1:
                dz = np.matmul(w_copy.T, dz) * (a_prev * (1 - a_prev))

    def train(
        self,
        X,
        Y,
        iterations=5000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100
    ):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        points = []
        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if verbose or graph:
                if i % step == 0 or i == iterations:
                    c = self.cost(Y, A)
                    points.append(i)
                    costs.append(c)
                    if verbose:
                        print("Cost after {} iterations: {}".format(i, c))

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(points, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance to a pickle file."""
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a DeepNeuralNetwork instance from a pickle file."""
        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as f:
            return pickle.load(f)
