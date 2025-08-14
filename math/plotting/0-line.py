#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    x = np.arange(0, 11)
    y = x ** 3
    plt.plot(x, y, color="red")  # color istifadə et
    plt.xlim(0, 10)
    plt.show()

line()
