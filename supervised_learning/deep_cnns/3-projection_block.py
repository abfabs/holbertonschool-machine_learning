#!/usr/bin/env python3
"""Projection block module."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block.

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12.
        s: stride of the first convolution in the main path and shortcut.

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    x = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    x = K.layers.Add()([x, shortcut])
    x = K.layers.Activation('relu')(x)

    return x
