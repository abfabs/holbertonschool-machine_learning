#!/usr/bin/env python3
"""
1-input.py
Builds a neural network using Keras functional API
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras (Functional API).
    - nx: number of input features
    - layers: list of node counts per layer
    - activations: list of activations per layer
    - lambtha: L2 regularization parameter
    - keep_prob: probability of keeping a node during dropout

    Returns: keras model
    """
    reg = K.regularizers.l2(lambtha)

    x_in = K.Input(shape=(nx,))
    x = x_in

    for i, (units, act) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            units=units,
            activation=act,
            kernel_regularizer=reg
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    return K.Model(inputs=x_in, outputs=x)
