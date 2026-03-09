#!/usr/bin/env python3
"""Transition layer module."""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer.

    Args:
        X: output from the previous layer.
        nb_filters: number of filters in X.
        compression: compression factor for the transition layer.

    Returns:
        The output of the transition layer and the number of filters within
        the output.
    """
    initializer = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)

    x = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(x)

    return x, nb_filters
