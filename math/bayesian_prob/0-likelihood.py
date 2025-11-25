#!/usr/bin/env python3
"""Likelihood"""
import numpy as np


def n_choose_k(n, k):
    """Calculate combination C(n, k) without using math module"""
    k = min(k, n - k)
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result

def likelihood(x, n, P):
  """This function calculates the likelihood of obtaining this data given 
  various hypothetical probabilities of developing severe side effects"""
  if not isinstance(n, int) or n<=0:
    raise ValueError("n must be a positive integer")
  if not isinstance(x, int) or x<0:
    raise ValueError("x must be an integer that is greater than or equal to 0") 
  if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
  if x > n:
        raise ValueError("x cannot be greater than n")
  if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

  coeff = n_choose_k(n, x)
  likelihoods = coeff * (P ** x) * ((1 - P) ** (n - x))

  return likelihoods
