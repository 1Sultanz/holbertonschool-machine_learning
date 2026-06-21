#!/usr/bin/env python3
"""This module includes the function
that creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """
    Args:
        array is the np.ndarray from which
         you should create the pd.DataFrame
        The columns of the pd.DataFrame should be
         labeled in alphabetical order and capitalized.
          There will not be more than 26 columns.
    Returns:
        the newly created pd.DataFrame
    """
    num_cols = array.shape[1]
    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=col_names)
