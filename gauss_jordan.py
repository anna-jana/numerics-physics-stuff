import fractions
import numpy as np

def show_matrix(B):
    matrix_print = []
    for i in range(B.shape[0]):
        row = "["
        for j in range(B.shape[1]):
            if B[i, j].denominator == 1:
                row += "%2d    " % (B[i, j].numerator,)
            else:
                row += "%2d/%2d " % (B[i, j].numerator, B[i, j].denominator)
        row += "]"
        matrix_print.append(row)
    return matrix_print

def show(A, B):
    print ""
    print_A = show_matrix(A)
    print_B = show_matrix(B)
    for row_A, row_B in zip(print_A, print_B):
        print row_A + " " + row_B

def map_map(f, xss):
    return [map(f, xs) for xs in xss]

A = np.array(map_map(fractions.Fraction,
                 [[6,3,4,5],
                  [1,2,2,1],
                  [2,4,3,2],
                  [3,3,4,2]]), dtype=fractions.Fraction)

def gauss_jordan(A):
    A = A.copy()

    num_rows, num_columns = A.shape

    # identity matrix with the same dimension as A
    E = np.array([[fractions.Fraction(1 if i == j else 0) for i in range(num_rows)] for j in range(num_columns)])

    # a "DSL" to invert the matrix
    def at(i, j):
        return A[i, j]

    def sub(i, j):
        A[i, :] -= A[j, :]
        E[i, :] -= E[j, :]

    def div(i, x):
        A[i, :] /= x
        E[i, :] /= x

    def swap_row(i, j):
        A[i, :], A[j, :] = A[j, :], A[i, :]
        E[i, :], E[j, :] = E[j, :], E[i, :]

    def get_non_zero_at_throu_row_swapping(index):
        for i in range(index, num_rows):
            if at(i, index) != 0:
                break
        else:
            raise ValueError("SINGULAR MATRIX")
        swap_row(index, i)

    # convert the matrix in a upper triangular matrix
    for i in range(0, num_rows):
        show(A, E)
        get_non_zero_at_throu_row_swapping(i)
        div(i, at(i, i))
        for j in range(i + 1, num_rows):
            if at(j, i) != 0:
                div(j, at(j, i))
                sub(j, i)

    # convert the upper triangular matrix into a identity matrix
    for i in range(num_rows - 1, -1, -1):
        show(A, E)
        div(i, at(i, i))
        for j in range(i - 1, -1, -1):
            if at(j, i) != 0:
                div(j, at(j, i))
                sub(j, i)

    show(A, E)

    return E

gauss_jordan(A)
