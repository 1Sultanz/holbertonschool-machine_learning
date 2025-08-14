#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)              # X oxu 0-dan 10-a qədər
    plt.plot(x, y, color="red")       # Qırmızı bərk xətt
    plt.xlim(0, 10)                    # X-ox diapazonu 0–10
    plt.show()

line()
