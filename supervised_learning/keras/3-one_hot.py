#!/usr/bin/env python3
"""
3-one_hot.py
Converts a label vector into a one-hot encoded matrix
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a vector of labels into a one-hot encoded matrix.

    labels is a numpy.ndarray of shape (m,) containing class labels
    classes is the total number of classes

    Returns: a one-hot encoded matrix of shape (m, classes)
    """
    if classes is None:
        return K.utils.to_categorical(labels)
    return K.utils.to_categorical(labels, classes)
