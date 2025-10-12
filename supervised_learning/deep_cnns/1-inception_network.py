#!/usr/bin/env python3
"""Identity Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """This function  builds the inception network as
     described in Going Deeper with Convolutions (2014)"""
    input_layer = K.Input(shape=(224, 224, 3))
    init = K.initializers.HeNormal(seed=None)

    conv = K.layers.Conv2D(
        filters=64, kernel_size=7, strides=2,
        padding="same", activation="relu",
        kernel_initializer=init
    )(input_layer)
    max_pooling = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same"
    )(conv)
    conv2 = K.layers.Conv2D(
        filters=64, kernel_size=1, strides=1,
        padding="same", activation="relu",
        kernel_initializer=init
    )(max_pooling)
    conv3 = K.layers.Conv2D(
        filters=192, kernel_size=3, strides=1,
        padding="same", activation="relu",
        kernel_initializer=init
    )(conv2)
    max_pooling2 = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same"
    )(conv3)
    inception_3a = inception_block(max_pooling2,
                                   [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a,
                                   [128, 128, 192, 32, 96, 64])
    max_pooling3 = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same"
    )(inception_3b)
    inception_4a = inception_block(max_pooling3,
                                   [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a,
                                   [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b,
                                   [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c,
                                   [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d,
                                   [256, 160, 320, 32, 128, 128])
    max_pooling4 = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same"
    )(inception_4e)
    inception_5a = inception_block(max_pooling4,
                                   [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a,
                                   [384, 192, 384, 48, 128, 128])
    avg_pool = K.layers.AveragePooling2D(
        pool_size=7, strides=1, padding="valid"
    )(inception_5b)
    drop = K.layers.Dropout(rate=0.4)(avg_pool)
    output_layer = K.layers.Dense(
        1000, activation="softmax",
        kernel_initializer=init
    )(drop)
    model = K.models.Model(input_layer, output_layer)
    return model
