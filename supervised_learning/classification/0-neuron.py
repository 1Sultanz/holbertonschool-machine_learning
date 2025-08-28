#!/usr/bin/env python3
"""Neuron"""

import numpy as np

class Neuron:
  """This class defines a single neuron performing
  binary classifiaction"""
  def __init__(self, nx):
    if type(nx) != int:
      raise TypeError("nx must be an integer")
    if nx < 1:
      raise ValueError("nx must be a positive integer")

    self.W = np.random.randn(nx + 1)
    self.b = 0
    self.A = 0
