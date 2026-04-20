#!/usr/bin/env python3
"""Module for performing forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l for forward
            propagation
        X: numpy.ndarray of shape (t, m, i) containing the input data
        h_0: numpy.ndarray of shape (l, m, h) containing initial hidden states

    Returns:
        H: numpy.ndarray of shape (t+1, l, m, h) containing all hidden states
        Y: numpy.ndarray of shape (t, m, o) containing all outputs
    """
    t, m, _ = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]

    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    outputs = []
    for step in range(t):
        layer_input = X[step]
        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(
                H[step, layer], layer_input)
            H[step + 1, layer] = h_next
            layer_input = h_next
        outputs.append(y)

    Y = np.array(outputs)
    return H, Y
