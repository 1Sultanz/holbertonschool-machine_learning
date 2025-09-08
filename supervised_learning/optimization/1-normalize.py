#!/usr/bin/env python3
"""Normalize"""
import numpy as np


def normalize(X, m, s):
    """This function normalizes a matrix"""
    X_std = (X - m) / s
    return X_std
