#!/usr/bin/env python3
"""Exponential Distribution"""


class Exponential:
    """This class represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1):
        """Class constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must conatain multiple values")
            self.lambtha = float(sum(data)/len(data))
