#!/usr/bin/env python3
"""Binomial Distribution"""


class Binomial:
    """This class represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """CLass Constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = float(1 - (variance / mean))
            self.n = round(mean / self.p)
            self.p = float(mean / self.n)

    @staticmethod
    def factorial(num):
        if num <= 1:
            return 1
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result

    def pmf(self, k):
        """Probability Mass Function for a given number of successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        binom = (self.factorial(self.n) /
                                (self.factorial(k) * self.factorial(self.n - k)))
        pmf = (binom * (self.p ** k)
                     * ((1 - self.p) ** (self.n - k)))
        return pmf

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            binom = self.factorial(self.n) / (self.factorial(i) * self.factorial(self.n - i))
            cdf += binom * (self.p ** i) * ((1 - self.p) ** (self.n - i))
        return cdf
