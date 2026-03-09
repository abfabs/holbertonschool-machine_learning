#!/usr/bin/env python3
"""Identity block module."""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block.

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3, F12.

    Returns:
        The activated output of the identity block.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    x = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
