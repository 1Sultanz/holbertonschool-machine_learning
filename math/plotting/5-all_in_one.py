#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def all_in_one():

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle("All in One")
    
    x0 = np.arange(0, 11)
    plt.subplot(3, 2, 1)
    plt.plot(x0, y0, 'r-')
    plt.axis((0, 10, None, None))
    
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, color='m', alpha=0.5)
    plt.xlabel('Height(in)')
    plt.ylabel('Weight(lbs)')
    plt.axis((60, 80, 170, 190))

    plt.subplot(3, 2, 3)
    plt.plot(x2, y2)
    plt.axis((0, 20000, None, None))
    plt.yscale('log')
    plt.xlabel('Time(years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, 'r--', x3, y32, 'g-')
    plt.axis((0, 20000, 0, 1))
    plt.xlabel('Time(years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.legend(['C-14', 'Ra-226'])

    fig.add_subplot(gs[2, :])
    bins = np.arange(0, 101, 10,)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xticks(bins)
    plt.axis((0, 100, 0, 30))
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

all_in_one()
