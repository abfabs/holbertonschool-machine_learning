#!/usr/bin/env python3
"""
0-sequential.py
Builds a neural network using Keras Sequential API
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras (Sequential).
    - nx: number of input features
    - layers: list of node counts per layer
    - activations: list of activations per layer
    - lambtha: L2 regularization parameter
    - keep_prob: probability of keeping a node during dropout

    Returns: keras model
    """
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)

    for i, (units, act) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(K.layers.Dense(
                units=units,
                activation=act,
                kernel_regularizer=reg,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units=units,
                activation=act,
                kernel_regularizer=reg
            ))

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
