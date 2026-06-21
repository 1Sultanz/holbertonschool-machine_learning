#!/usr/bin/env python3
"""This module contains python script that
creates a pd.DataFrame from a dictionary"""
import pandas as pd

"""
Instructions:
    The first column should be labeled First and
     have the values 0.0, 0.5, 1.0, and 1.5
    The second column should be labeled Second and
     have the values one, two, three, four
    The rows should be labeled A, B, C, and D, respectively
    The pd.DataFrame should be saved into the variable df
"""

from_dict = {"First": [0.0, 0.5, 1.0, 1.5],
             "Second": ['one', 'two', 'three', 'four']}

row_names = ['A', 'B', 'C', 'D']

df = pd.DataFrame(from_dict, index=row_names)
