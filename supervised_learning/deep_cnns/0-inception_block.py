#!/usr/bin/env python3
"""Inception block module."""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block.

    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing:
            F1, F3R, F3, F5R, F5, FPP

    Returns:
        The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv_3x3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv_3x3_reduce)

    conv_5x5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv_5x5_reduce)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(pool)

    output = K.layers.Concatenate()(
        [conv_1x1, conv_3x3, conv_5x5, pool_proj]
    )

    return output
