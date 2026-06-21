#!/usr/bin/env python3
"""This module includes the function that takes
a pd.DataFrame and Sorts the data in reverse chronological order."""


def flip_switch(df):
    """Transposes the sorted dataframe.
    Returns: the transformed pd.DataFrame."""
    df = df.sort_values(by="Timestamp", ascending=False)
    return df.transpose()
