#!/usr/bin/env python3
"""
12-test.py
Tests a keras model
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    network is the network model to test
    data is the input data to test the model with
    labels are the correct one-hot labels of data
    verbose determines if output is printed during testing

    Returns: the loss and accuracy of the model, respectively
    """
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
