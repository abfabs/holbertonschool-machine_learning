#!/usr/bin/env python3
"""Module for LSTMCell class representing an LSTM unit."""
import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) unit."""

    def __init__(self, i, h, o):
        """
        Initialize the LSTMCell.

        Args:
            i: dimensionality of the data input
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            c_prev: numpy.ndarray of shape (m, h) with previous cell state
            x_t: numpy.ndarray of shape (m, i) with input data for the cell

        Returns:
            h_next: next hidden state, shape (m, h)
            c_next: next cell state, shape (m, h)
            y: output of the cell after softmax, shape (m, o)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self._sigmoid(np.dot(concat, self.Wf) + self.bf)
        u = self._sigmoid(np.dot(concat, self.Wu) + self.bu)
        c_candidate = np.tanh(np.dot(concat, self.Wc) + self.bc)
        o = self._sigmoid(np.dot(concat, self.Wo) + self.bo)

        c_next = f * c_prev + u * c_candidate
        h_next = o * np.tanh(c_next)

        logits = np.dot(h_next, self.Wy) + self.by
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        return h_next, c_next, y

    def _sigmoid(self, x):
        """Apply sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
