#!/usr/bin/env python3
"""
13-predict.py
Makes predictions using a keras model
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose determines if output is printed during prediction

    Returns: the prediction for the data
    """
    return network.predict(
        x=data,
        verbose=verbose
    )
