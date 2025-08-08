#!/usr/bin/env python3
"""Line up"""


def add_arrays(arr1, arr2):
    """This function adds two arrays element-wise"""
    arr = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr.append(arr1[i] + arr2[i])
        return arr
    else:
        None
 

if __name__ == "__main__":
    arr1 = [1, 2, 3, 4]
    arr2 = [5, 6, 7, 8]
    print(add_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
    print(add_arrays(arr1, [1, 2, 3]))
