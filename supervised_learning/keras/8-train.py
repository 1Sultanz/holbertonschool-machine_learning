#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False
                ):
    """This function trains a model using mini-batch gradient descent"""
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
        ))
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            return alpha/(1 + decay_rate*epoch)
    if filepath:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath, monitor='val_loss', verbose=1,
            save_best_only=save_best
        ))

    callbacks.append(K.callbacks.LearningRateScheduler(
        scheduler, verbose=1
    ))

    history = network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, validation_data=validation_data,
        verbose=verbose, shuffle=shuffle, callbacks=callbacks
    )
    return history
