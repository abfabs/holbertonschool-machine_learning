#!/usr/bin/env python3
"""Transfer learning on CIFAR-10 using a Keras application."""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-processes the CIFAR-10 data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) with image data.
        Y: numpy.ndarray of shape (m,) or (m, 1) with labels.

    Returns:
        X_p: preprocessed image data.
        Y_p: one-hot encoded labels.
    """
    X_p = K.applications.mobilenet_v2.preprocess_input(X.astype("float32"))
    Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


def build_head(input_shape):
    """Builds the classifier head.

    Args:
        input_shape: shape of the feature maps from the base model.

    Returns:
        A keras model representing the classifier head.
    """
    inputs = K.Input(shape=input_shape)
    x = K.layers.GlobalAveragePooling2D()(inputs)
    x = K.layers.Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal"
    )(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer="he_normal"
    )(x)
    return K.models.Model(inputs=inputs, outputs=outputs)


def build_models():
    """Builds the feature extractor, classifier head, and full model.

    Returns:
        feature_extractor: model used to cache frozen features.
        head: classifier head model.
        full_model: full end-to-end model.
        base_model: pretrained application model.
    """
    inputs = K.Input(shape=(32, 32, 3))

    x = K.layers.Lambda(
        lambda img: K.layers.Resizing(96, 96)(img)
    )(inputs)

    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(96, 96, 3)
    )
    base_model.trainable = False

    features = base_model(x, training=False)
    head = build_head(base_model.output_shape[1:])
    outputs = head(features)

    feature_extractor = K.models.Model(inputs=inputs, outputs=features)
    full_model = K.models.Model(inputs=inputs, outputs=outputs)

    return feature_extractor, head, full_model, base_model


def train_model():
    """Loads CIFAR-10, trains the model, and saves it as cifar10.h5."""
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    X_val = X_train[-5000:]
    Y_val = Y_train[-5000:]
    X_train = X_train[:-5000]
    Y_train = Y_train[:-5000]

    feature_extractor, head, model, base_model = build_models()

    train_features = feature_extractor.predict(
        X_train,
        batch_size=128,
        verbose=1
    )
    val_features = feature_extractor.predict(
        X_val,
        batch_size=128,
        verbose=1
    )

    head.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]

    head.fit(
        train_features,
        Y_train,
        validation_data=(val_features, Y_val),
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        if isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    model.save("cifar10.h5")


if __name__ == "__main__":
    train_model()
