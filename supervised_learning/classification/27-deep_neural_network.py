#!/usr/bin/env python3
"""
27-deep_neural_network.py

Defines a DeepNeuralNetwork class for multiclass classification.
Implements forward propagation (softmax output), cost (cross-entropy),
and evaluation (one-hot predictions), plus persistence.
"""

import numpy as np
import os
import pickle


class DeepNeuralNetwork:
    """
    Deep neural network for multiclass classification.

    Hidden layers use sigmoid activation.
    The output layer uses softmax activation.
    """

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of positive integers representing
                the number of nodes in each layer.

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
        """
        Get the number of layers in the network.

        Returns:
            int: Number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Get the cache of intermediary values.

        Returns:
            dict: Cached activations.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Get the weights and biases of the network.

        Returns:
            dict: Weights and biases.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation for multiclass classification.

        Stores activations A0..AL in the cache:
            A0 is X
            A1..A(L-1) are sigmoid activations
            AL is softmax activation

        Args:
            X (np.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (A_L, cache)
                A_L is the softmax output of the last layer.
                cache contains all intermediary activations.
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            w = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            a_prev = self.__cache["A{}".format(layer - 1)]

            z = np.matmul(w, a_prev) + b

            if layer == self.__L:
                z_shift = z - np.max(z, axis=0, keepdims=True)
                exp_z = np.exp(z_shift)
                a = exp_z / np.sum(exp_z, axis=0, keepdims=True)
            else:
                a = 1 / (1 + np.exp(-z))

            self.__cache["A{}".format(layer)] = a

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Compute the multiclass cross-entropy cost.

        Y is one-hot of shape (classes, m).
        A is softmax output of shape (classes, m).

        Args:
            Y (np.ndarray): One-hot correct labels (classes, m).
            A (np.ndarray): Softmax probabilities (classes, m).

        Returns:
            float: Computed cost.
        """
        m = Y.shape[1]
        a_safe = np.clip(A, 1e-8, 1.0)
        loss = Y * np.log(a_safe)
        return float((-1 / m) * np.sum(loss))

    def evaluate(self, X, Y):
        """
        Evaluate the network's predictions for multiclass classification.

        Returns one-hot predictions with shape (classes, m):
        prediction[class_idx, example_idx] = 1 at the argmax class.

        Args:
            X (np.ndarray): Input data of shape (nx, m).
            Y (np.ndarray): One-hot correct labels of shape (classes, m).

        Returns:
            tuple: (prediction, cost)
                prediction is a one-hot array (classes, m)
                cost is the cross-entropy cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        pred = np.zeros_like(A)
        idx = np.argmax(A, axis=0)
        pred[idx, np.arange(A.shape[1])] = 1

        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent.

        For softmax + cross-entropy, dZ_L = A_L - Y, which matches the
        existing backprop structure from previous tasks.

        Args:
            Y (np.ndarray): One-hot correct labels of shape (classes, m).
            cache (dict): Cached activations from forward propagation.
            alpha (float): Learning rate.
        """
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
        """
        Train the neural network.

        Args:
            X (np.ndarray): Input data of shape (nx, m).
            Y (np.ndarray): One-hot labels of shape (classes, m).
            iterations (int): Number of training iterations.
            alpha (float): Learning rate.
            verbose (bool): Print cost during training.
            graph (bool): Plot training cost.
            step (int): Step size for printing/plotting.

        Returns:
            tuple: Evaluation of training data (prediction, cost).
        """
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
                        print(
                            "Cost after {} iterations: {}".format(i, c)
                        )

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
        """
        Save the instance to a pickle file.

        Args:
            filename (str): File path to save the object.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a DeepNeuralNetwork instance from a file.

        Args:
            filename (str): Path to the pickle file.

        Returns:
            DeepNeuralNetwork or None: Loaded object or None if file
            does not exist.
        """
        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as f:
            return pickle.load(f)
