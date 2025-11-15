#!/usr/bin/env python3
"""Normal Distribution"""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """This class represents a normal distribution"""

    def __init__(self, data=None, mean=0, stddev=1):
        """class constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Normalize Normal"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """X value"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Probabilit Density Function"""
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * \
            e ** (-0.5 * self.z_score(x) ** 2)

    @staticmethod
    def error_f(x):
        return ((2 / (pi) ** 0.5) *
                (x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216))

    def cdf(self, x):
        """Cumulative Distribution Function"""
        return 0.5 * (1 + self.error_f(self.z_score(x) / (2 ** 0.5)))
