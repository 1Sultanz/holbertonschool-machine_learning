#!/usr/bin/env python3
"""Plotting."""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot a line where x axis is numbers 0-10 and y is x**3."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')
    plt.axis((0, 10, None, None))
    plt.show()
