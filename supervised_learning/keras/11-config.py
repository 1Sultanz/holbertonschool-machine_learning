#!/usr/bin/env python3
"""Tensorflow2 & Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """Save Keras model architecture to JSON file."""
    with open(filename, 'w') as f:
        f.write(network.to_json())

def load_config(filename):
    """Load Keras model architecture from JSON file."""
    with open(filename, 'r') as f:
        model_json = f.read()
    return model_from_json(model_json)
