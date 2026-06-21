#!/usr/bin/env python3
"""This module includes the function that
takes a pd.DataFrame as input and performs the following"""


def array(df):
    """
    Args:
        df is a pd.DataFrame containing columns named High and Close.
        The function should select the last 10 rows of the
         High and Close columns.
        Convert these selected values into a numpy.ndarray.
    Returns:
        the numpy.ndarray
    """
    portion = df[["High", "Close"]].tail(10)
    return portion.to_numpy()
