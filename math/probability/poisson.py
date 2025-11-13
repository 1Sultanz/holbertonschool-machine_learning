#!/usr/bin/env python3
"""Initialize Poisson"""


class poisson:
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
            if data < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
