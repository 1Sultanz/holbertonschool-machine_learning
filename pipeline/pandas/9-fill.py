#!/usr/bin/env python3
"""This module includes function that takes
a pd.DataFrame and Removes the Weighted_Price column."""


def fill(df):
    """
    Fills missing values in the Close column
     with the previous row’s value.
    Fills missing values in the High, Low, and Open columns
     with the corresponding Close value in the same row.
    Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
    Returns:
        the modified pd.DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].ffill()
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[["Volume_(BTC)",
                                                    "Volume_(Currency)"
                                                    ]].fillna(0)
    return df
