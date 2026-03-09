#!/usr/bin/env python3
"""DenseNet-121 module."""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture.

    Args:
        growth_rate: growth rate of the dense blocks.
        compression: compression factor of transition layers.

    Returns:
        The keras model.
    """
    initializer = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(x)

    x, nb_filters = dense_block(x, 2 * growth_rate, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(7, 7),
        padding='same'
    )(x)

    Y = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(x)

    model = K.models.Model(inputs=X, outputs=Y)
    return model
