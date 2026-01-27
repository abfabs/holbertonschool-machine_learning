#!/usr/bin/env python3
"""
5-train.py
Trains a keras model using mini-batch gradient descent with optional validation
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels is a one-hot numpy.ndarray of shape (m, classes) containing labels
    batch_size is the batch size
    epochs is the number of epochs
    validation_data is the data to validate the model with, if not None
    verbose determines if training output is printed
    shuffle determines whether to shuffle the data each epoch

    Returns: the History object generated after training the model
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )
