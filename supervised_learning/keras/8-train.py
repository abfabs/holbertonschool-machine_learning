#!/usr/bin/env python3
"""
8-train.py
Trains a keras model using mini-batch gradient descent with 
optional validation, early stopping, learning rate decay, 
and saving the best model
"""

import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    early_stopping=False,
    patience=0,
    learning_rate_decay=False,
    alpha=0.1,
    decay_rate=1,
    save_best=False,
    filepath=None,
    verbose=True,
    shuffle=False
):
    """
    Trains a model using mini-batch gradient descent.

    network is the model to train
    data is a numpy.ndarray of shape (m, nx)
    labels is a one-hot numpy.ndarray of shape (m, classes)
    batch_size is the batch size
    epochs is the number of epochs
    validation_data is the data to validate the model with
    early_stopping indicates whether early stopping should be used
    patience is the patience used for early stopping
    learning_rate_decay indicates whether learning rate decay is used
    alpha is the initial learning rate
    decay_rate is the decay rate
    save_best indicates whether to save the best model
    filepath is the path where the model is saved
    verbose determines if training output is printed
    shuffle determines whether to shuffle the data each epoch

    Returns: the History object generated after training the model
    """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(
                K.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience
                )
            )

        if learning_rate_decay:
            def schedule(epoch):
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(
                K.callbacks.LearningRateScheduler(
                    schedule,
                    verbose=1
                )
            )

        if save_best:
            callbacks.append(
                K.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    monitor='val_loss',
                    save_best_only=True
                )
            )

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
