#!/usr/bin/env python3
"""F1 - Score"""
import numpy as np


def f1_score(confusion):
    """This function calculates the F1 score
     of a confusion matrix"""
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    prec = precision(confusion)
    sens = sensitivity(confusion)
    classes = confusion.shape[0]
    f1 = np.zeros(classes)

    for i in range(classes):
        if prec[i] + sens[i] == 0:
            f1[i] = 0
        else:
            f1[i] = (2 * prec[i] + sens[i]) / (
                    prec[i] + sens[i])
    return f1
