#!/usr/bin/env python3
"""This module includes the function that takes
two pd.DataFrame objects and Rearranges the MultiIndex
so that Timestamp is the first level."""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates the bits-tamp and coinbase tables
     from timestamps 1417411980 to 1417417980, inclusive.
    Adds keys to the data, labeling rows from df2
     as bits-tamp and rows from df1 as coinbase.
    Ensures the data is displayed in chronological order.
    Returns: the concatenated pd.DataFrame.
    """
    df1_tp, df2_tp = index(df1), index(df2)
    df1_partition = df1_tp.loc[1417411980:1417417980]
    df2_partition = df2_tp.loc[1417411980:1417417980]
    concatenated = pd.concat(
        [df2_partition, df1_partition],
        keys=["bitstamp", "coinbase"]
    )

    # Swap levels: change (Source, Timestamp) to (Timestamp, Source)
    # Sort the index to ensure 1417411980 comes before 1417412040
    result = concatenated.swaplevel(0, 1).sort_index()

    return result
