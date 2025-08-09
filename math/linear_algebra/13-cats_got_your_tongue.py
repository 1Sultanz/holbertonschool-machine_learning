#!/usr/bin/env python3
"""Cat's Got Your Tongue"""
def np_cat(mat1, mat2, axis=0):
    """This function concatenates two matrices along a specific axis"""
    return np.cancatenate((mat1, mat2), axis)
