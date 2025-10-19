#!/usr/bin/env python3
"""Transfer Learning"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """Pre-processes the data for the model"""
    X_p = K.applications.resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    input_tensor = K.layers.Input(shape=(32, 32, 3))

    resize_layer = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224))
    )(input_tensor)

    base_model = K.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=resize_layer,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    output_tensor = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    datagen.fit(x_train)

    print("--- Training the new classification head ---")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=5,
        validation_data=(x_test, y_test)
    )

    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_cb = K.callbacks.ModelCheckpoint(
        'cifar10.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
    early_stopping_cb = K.callbacks.EarlyStopping(
        patience=5,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    print("\n--- Fine-tuning the model ---")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    print("\n--- Saving final model to cifar10.h5 ---")
    model.save('cifar10.h5')
