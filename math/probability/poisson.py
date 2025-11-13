#!/usr/bin/env python3
"""Initialize Poisson"""


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
        factorial = 1
        for i in range(1, k+1):
            factorial *= i
        pmf = ((e ** -self.lambtha) * self.lambtha ** k) / factorial
        return pmf

    def cdf(self, k):
        """Cumulative Distribution Function: P(X <= k)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k+1):
            cdf += self.pmf(i)
        return cdf
