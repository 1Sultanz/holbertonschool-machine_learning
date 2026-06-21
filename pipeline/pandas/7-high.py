#!/usr/bin/env python3
"""This module includes the function that takes
a pd.DataFrame and sorts it by the High price in descending order."""


def high(df):
    """
    Returns: the sorted pd.DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
