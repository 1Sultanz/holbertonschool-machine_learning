#!/usr/bin/env python3
"""Precision"""
import numpy as np


def precision(confusion):
    """This function  calculates the precision
     for each class in a confusion matrix"""
    classes = confusion.shape[0]
    precisions = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i][i]
        FP = np.sum(confusion[:, i]) - TP
        if TP + FP == 0:
            precisions[i] = 0
        else:
            precisions[i] = TP / (TP + FP)
    return precisions
