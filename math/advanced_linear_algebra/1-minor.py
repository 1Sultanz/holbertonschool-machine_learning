#!/usr/bin/env python3
"""Minor"""


def determinant(matrix):
    """This function calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for x in range(len(matrix)):
        sub_matrix = [row[:x] + row[x + 1:] for row in matrix[1:]]
        det += ((-1) ** x) * matrix[0][x] * determinant(sub_matrix)

    return det


def minor(matrix):
    """This function calculates the minor of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(sub_list):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor = []
    for r in range(len(matrix)):
        i = []
        for c in range(len(matrix[0])):
            matrix_copy = [x[:] for x in matrix]
            del matrix_copy[r]
            for row in matrix_copy:
                del row[c]
            i.append(determinant(matrix_copy))
        minor.append(i)

    return minor
