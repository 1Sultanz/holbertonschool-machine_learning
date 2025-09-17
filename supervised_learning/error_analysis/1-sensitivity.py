#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """This function calculates the sensitivity
    for each class in a confusion matrix"""
    classes = confusion.shape[0]
    sensitivities = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i][i]
        FN = np.sum(confusion[i, :]) - TP
        if TP + FN == 0:
            sensitivities[i] = 0
        else:
            sensitivities[i] = TP / (TP + FN)
    return sensitivities
