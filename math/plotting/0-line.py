#!/usr/bin/env python3
"""Line Graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Source code to plot y as a line graph"""
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)
    plt.plot(x, y, color="red")
    plt.xlim(0, 10)
    plt.show()


if __name__ == "__main__":
    line()
