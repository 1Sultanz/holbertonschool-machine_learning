#!/usr/bin/env python3
"""This module includes the function that
takes a pd.DataFrame as input and performs the following"""
import pandas as pd


def rename(df):
    """
    Args:
        df is a pd.DataFrame containing a column named Timestamp.
        The function should rename the Timestamp column to Datetime.
        Convert the timestamp values to datatime values
        Display only the Datetime and Close column
    Returns:
        the modified pd.DataFrame
    """
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Converting to datetime from timestamp using 's'
    # because default data is in nanoseconds while we want seconds
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')

    # Filtering to display only Datetime and Close columns
    df = df[["Datetime", "Close"]]
    return df
