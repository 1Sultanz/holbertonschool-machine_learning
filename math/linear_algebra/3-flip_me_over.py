#!/usr/bin/env python3
"""Flip me over"""


def matrix_transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))] 



if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    print(mat1)
    print(matrix_transpose(mat1))
    mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    print(mat2)
    print(matrix_transpose(mat2))
