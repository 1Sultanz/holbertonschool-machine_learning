#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""

def poly_derivative(poly):
    """this function calculates the derivative of a polynomial"""
    if type(poly) != list:
      return None
    if len(poly) == 1:
      return [0]
    if len(poly) == 0:
        return None
    for element in poly:
      if type(element) != int and type(element) != float:
        return None
    if poly[0] == 0:
      return [0]
    
    derivative = [poly[i] * i for i in range(1, len(poly))]

    if all(coef == 0 for coef in derivative):
          return [0]

    return derivative
