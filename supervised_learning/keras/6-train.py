#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """This function trains a model using mini-batch gradient descent"""
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
        ))

    history = network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, validation_data=validation_data,
        verbose=verbose, shuffle=shuffle, callbacks=callbacks
    )
    return history
