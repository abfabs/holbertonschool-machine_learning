#!/usr/bin/env python3
"""Module for GRUCell class representing a Gated Recurrent Unit cell."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """
        Initialize the GRUCell.

        Args:
            i: dimensionality of the data input
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with input data for the cell

        Returns:
            h_next: next hidden state, shape (m, h)
            y: output of the cell after softmax, shape (m, o)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = self._sigmoid(np.dot(concat, self.Wr) + self.br)

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_candidate = np.tanh(np.dot(concat_r, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_candidate

        logits = np.dot(h_next, self.Wy) + self.by
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        return h_next, y

    def _sigmoid(self, x):
        """Apply sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
