#!/usr/bin/env python3
"""LeNet-5 (Keras)"""
from tensorflow import keras as K


def lenet5(X):
    """This function  builds a modified version
    of the LeNet-5 architecture using keras"""
    initialize = K.initializers.HeNormal(seed=0)
    model = K.Sequential([
        K.layers.Conv2D(
            filters=6,
            kernel_size=5,
            padding="same",
            kernel_initializer=initialize,
            activation="relu"
        ),
        K.layers.MaxPooling2D(
            pool_size=2,
            strides=2
        ),
        K.layers.Conv2D(
            filters=16,
            kernel_size=5,
            padding="valid",
            kernel_initializer=initialize,
            activation="relu"
        ),
        K.layers.MaxPooling2D(
            pool_size=2,
            strides=2
        ),
        K.layers.Flatten(),
        K.layers.Dense(120, activation="relu", kernel_initializer=initialize),
        K.layers.Dense(84, activation="relu", kernel_initializer=initialize),
        K.layers.Dense(10, activation="softmax", kernel_initializer=initialize)
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=K.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model
