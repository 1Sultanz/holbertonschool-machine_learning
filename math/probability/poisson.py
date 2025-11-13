#!/usr/bin/env python3
"""Initialize Poisson"""
import numpy as np
import math


class Poisson:
    """This class represent a poisson distribution"""
    def __init__(self, data=None, lambtha=1):
        """Class Constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Probability Mass Function: P(X = k)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        pmf = ((e ** -self.lambtha) * self.lambtha ** k) / math.factorial(k)
        return pmf
