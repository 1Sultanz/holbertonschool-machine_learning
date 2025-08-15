#!/usr/bin/env python3
"""Change of scale"""
import numpy as np
import matplotlib.pyplot as plt

def two():
    """Source code to plot x ↦ y as a line graph"""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, color="red", linestyle="dashed")
    plt.plot(x, y2, color="green")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.axis((0, 20000, 0, 1))
    plt.legend(["C-14","Ra-226"])
    plt.show()
