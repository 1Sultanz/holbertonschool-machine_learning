#!/usr/bin/env python3
"""Normal Distribution"""


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
            self.mean = float(sum(data)/len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Normalize Normal"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """X value"""
        return z * self.stddev + self.mean
