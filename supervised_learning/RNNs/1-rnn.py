#!/usr/bin/env python3
"""Module for performing forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell used for forward propagation
        X: numpy.ndarray of shape (t, m, i) containing the input data
        h_0: numpy.ndarray of shape (m, h) containing the initial hidden state

    Returns:
        H: numpy.ndarray of shape (t+1, m, h) containing all hidden states
        Y: numpy.ndarray of shape (t, m, o) containing all outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    outputs = []
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        outputs.append(y)

    Y = np.array(outputs)
    return H, Y
