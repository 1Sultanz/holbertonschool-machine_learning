#!/usr/bin/env python3
"""This module contains the function that
takes a pd.DataFrame and Extracts the columns
High, Low, Close, and Volume_BTC."""


def slice(df):
    """Selects every 60th row from these columns.
    Returns: the sliced pd.DataFrame"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
