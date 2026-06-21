#!/usr/bin/env python3
"""This module contains the function that takes
two pd.DataFrame objects and Indexes both dataframes
on their Timestamp columns."""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Includes all timestamps from df2 (bits-tamp)
     up to and including timestamp 1417411920.
    Concatenates the selected rows from df2
     to the top of df1 (coinbase).
    Adds keys to the concatenated data, labeling the rows
     from df2 as bits-tamp and the rows from df1 as coinbase.
    Returns the concatenated pd.DataFrame.
    """
    df1_tp, df2_tp = index(df1), index(df2)
    df2_partition = df2_tp.loc[:1417411920]
    concatenated = pd.concat(
        [df2_partition, df1_tp],
        keys=["bitstamp", "coinbase"]
    )

    return concatenated
