#!/usr/bin/env python3
"""5-lenet5.py
Builds a modified LeNet-5 architecture using Keras.
"""

from tensorflow import keras as K


def lenet5(X):
    """Builds a modified LeNet-5 model.

    Architecture:
        Conv2D(6, 5x5, same) -> ReLU
        MaxPool(2x2, 2x2)
        Conv2D(16, 5x5, valid) -> ReLU
        MaxPool(2x2, 2x2)
        Flatten
        Dense(120) -> ReLU
        Dense(84) -> ReLU
        Dense(10) -> Softmax

    All layers with weights use he_normal initialization with seed=0.

    Args:
        X (K.Input): Input tensor of shape (None, 28, 28, 1)

    Returns:
        K.Model: Compiled Keras model (Adam optimizer, accuracy metric)
    """
    he_init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=he_init,
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=he_init,
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv2)

    flat = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        units=120,
        activation="relu",
        kernel_initializer=he_init,
    )(flat)

    fc2 = K.layers.Dense(
        units=84,
        activation="relu",
        kernel_initializer=he_init,
    )(fc1)

    y = K.layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer=he_init,
    )(fc2)

    model = K.Model(inputs=X, outputs=y)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
