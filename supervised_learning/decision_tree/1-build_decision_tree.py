#!/usr/bin/env python3
"""Decision tree implementation."""

import numpy as np


class Node:
    """Tree node that splits data based on a feature and threshold."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        """Create a node with feature, threshold, children and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return maximum depth of this node and its children."""
        if self.right_child is None and self.left_child is None:
            return self.depth

        max_depth = self.depth

        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below including this one (or only leaves)."""
        count = 0
        if not only_leaves or self.is_leaf:
            count += 1
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)
        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        return count


class Leaf(Node):
    """Leaf node that stores a prediction value."""

    def __init__(self, value, depth=None):
        """Create a leaf with value and depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth


class Decision_Tree:
    """Decision tree that builds from nodes and leaves."""

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        """Create a tree with max depth, min samples, seed and split rule."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)
