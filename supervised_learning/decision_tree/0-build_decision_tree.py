#!/usr/bin/env python3

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self) :
      """This function returns max depth of tree"""
      if self.right_child is None and self.left_child is None:
        return self.depth
      max_depth = self.depth
      if self.right_child is not None:
        max_depth = max(max_depth, self.right_child.max_depth_below())
      if self.left_child is not None:
        max_depth = max(max_depth, self.left_child.max_depth_below())
      return max_depth

class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        return self.depth
