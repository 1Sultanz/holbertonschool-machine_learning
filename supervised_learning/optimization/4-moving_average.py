#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def moving_average(data, beta):
    """Moving Average"""
    averages = []
    v = 0  # moving average value
    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x  # update moving avg
        corrected = v / (1 - beta ** t)  # bias correction
        averages.append(corrected)
    return averages
