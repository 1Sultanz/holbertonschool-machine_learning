#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""

def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(element, (int, float)) for element in poly):
        return None

    if len(poly) == 1:
        return [0]

    return [poly[i] * i for i in range(1, len(poly))]
