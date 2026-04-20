#!/usr/bin/env python3
"""Module for RNNCell class representing a simple RNN cell."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple Recurrent Neural Network."""

    def __init__(self, i, h, o):
        """
        Initialize the RNNCell.

        Args:
            i: dimensionality of the data input
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
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
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        z = np.dot(h_next, self.Wy) + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return h_next, y
