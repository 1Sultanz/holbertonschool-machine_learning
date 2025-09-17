#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """This function calculates the specificity
     for each class in a confusion matrix"""
    classes = confusion.shape[0]
    spec = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i][i]
        FN = np.sum(confusion[i, :]) - TP
        FP = np.sum(confusion[:, i]) - TP
        TN = np.sum(confusion) - (TP + FN + FP)
        if TN + FP == 0:
            spec[i] = 0
        else:
            spec[i] = TN / (TN + FP)
    return spec
