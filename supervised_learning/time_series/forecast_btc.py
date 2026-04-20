#!/usr/bin/env python3
"""
Trains and validates a Keras RNN model to forecast BTC closing price.
Uses preprocessed data from preprocess_data.py.
Predicts BTC close price at the next hour given the past 24 hours.
"""
import numpy as np
import tensorflow as tf


def load_data():
    """
    Load preprocessed BTC sequences from disk.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    X = np.load('X_btc.npy')
    y = np.load('y_btc.npy')

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    return X_train, y_train, X_val, y_val


def make_dataset(X, y, batch_size=64, shuffle=False):
    """
    Create a tf.data.Dataset from numpy arrays.

    Args:
        X: input sequences array of shape (n, window, features)
        y: target values array of shape (n,)
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset

    Returns:
        Batched tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(window=1440, features=6):
    """
    Build an LSTM-based RNN model for BTC price forecasting.

    Args:
        window: number of time steps in each input sequence
        features: number of input features per time step

    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


if __name__ == '__main__':
    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    window = X_train.shape[1]
    features = X_train.shape[2]

    print("Building model...")
    model = build_model(window=window, features=features)
    model.summary()

    train_ds = make_dataset(X_train, y_train, batch_size=64, shuffle=True)
    val_ds = make_dataset(X_val, y_val, batch_size=64, shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )

    val_loss, val_mae = model.evaluate(val_ds)
    print(f"Validation MSE: {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")

    model.save('btc_forecast_model.keras')
    print("Model saved to btc_forecast_model.keras")
