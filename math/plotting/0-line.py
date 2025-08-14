#!/usr/bin/env python3
"""
Module that plots y = x^3 as a solid red line with x-axis from 0 to 10
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y = x^3 as a red line, with the x-axis ranging from 0 to 10
    """
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)
    plt.plot(x, y, color="red")
    plt.xlim(0, 10)
    plt.show()


if __name__ == "__main__":
    line()
