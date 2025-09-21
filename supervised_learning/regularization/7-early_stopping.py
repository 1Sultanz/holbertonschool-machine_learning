#!/usr/bin/env python3
""""Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """This function determines if you
     should stop gradient descent early"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
