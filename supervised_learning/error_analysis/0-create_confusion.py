#!/usr/bin/env python3
"""Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """This function creates a confusion matrix"""
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes))

    for t, p in zip(true_classes, pred_classes):
        confusion_matrix[t][p] += 1
    return confusion_matrix
