#!/usr/bin/env python3
"""This module includes the function that takes
a pd.DataFrame and Computes descriptive statistics
for all columns except the Timestamp column."""


def analyze(df):
    """
    Returns a new pd.DataFrame containing these statistics.
    """
    desc = list(df.columns)
    return df[desc[1:]].describe()
