#!/usr/bin/env python3
"""Our life is the sum total of all the decisions we make every day, 
    and those decisions are determined by our priorities"""


def summation_i_squared(n):
    """Sum total"""
    if type(n) is not int or n < 1:
        return None
    else:
        return (n*(n+1)*(n+2)/6)
