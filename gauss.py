import numpy as np
from scipy.linalg import solve

def gauss_elim(A, b):
    # don't destroy the old arrays
    A = A.copy()
    b = b.copy()
    # there are n equations and variables
    n = b.size
    eq_num = n
    var_num = n
    # the equations are the rows and each column is a variable
    # eliminate all variables i in 0..n-1 in the equations i+1..n-1
    for var_index_to_elim in xrange(0, var_num):
        # find the pivot (max abs) in the row below (and including the main diagonal
        avl = A[var_index_to_elim:, var_index_to_elim]
        best_index = np.argmax(np.abs(avl))
        best_index_in_A = var_index_to_elim + best_index
        # if no pivot != 0 was found fail
        best_pivot_val = A[best_index_in_A, var_index_to_elim]
        if best_pivot_val == 0.0:
            raise ValueError("SINGULAR MATRIX")
        # swap pivot in A and b
        best_row = A[best_index_in_A, :]
        original_row = A[var_index_to_elim, :]
        A[best_index_in_A, :] = best_row
        A[var_index_to_elim, :] = original_row
        # now is var_index_to_elim also the index of the equation to elimintate with
        eq_index_to_elim_with = var_index_to_elim
        # eliminate the variable `var_index_to_elim` in the equations var_index_to_elim+1..n-1
        for eq_index_to_elim_in in xrange(var_index_to_elim + 1, eq_num):
            # elimintate the variable var_index_to_elim in eq_index_to_elim_in
            # I: equation to eliminate with = A[eq_index_to_elim_with, :]
            eq_to_elim_with = A[eq_index_to_elim_with, :]
            # II: equation to eliminate in = A[eq_index_to_elim_in, :]
            eq_to_elim_in = A[eq_index_to_elim_in, :]
            # the variable to use to elimintate with is at I[var_index_to_elim] =
            # A[eq_index_to_elim_with, var_index_to_elim] = A[var_index_to_elim, var_index_to_elim]
            # = coeff_to_elim_with
            coeff_to_elim_with = eq_to_elim_with[var_index_to_elim]
            # the variable to elimintate is at II[var_index_to_elim] = coeff_to_elim =
            # A[eq_index_to_elim_in, var_index_to_elim]
            coeff_to_elim = eq_to_elim_in[var_index_to_elim]
            # alpha = coeff_to_elim/coeff_to_elim_with
            alpha = coeff_to_elim/coeff_to_elim_with
            # II <- II - alpha*I
            eliminated_eq = eq_to_elim_in - alpha*eq_to_elim_with
            # but the equation back
            A[eq_index_to_elim_in, :] = eliminated_eq
            # b[eq_index_to_elim_in] -= b[eq_index_to_elim_with]*alpha
            b[eq_index_to_elim_in] -= alpha*b[eq_index_to_elim_with]
    return A, b


def backsubst(A, b):
    n = b.size
    x = np.zeros(n)
    for var_to_solve in xrange(n - 1, -1, -1):
        other_subs = 0.0
        for var_to_sub in xrange(var_to_solve + 1, n):
            other_subs += x[var_to_sub]*A[var_to_solve, var_to_sub]
        x[var_to_solve] = (b[var_to_solve] - other_subs)/A[var_to_solve, var_to_solve]
    return x

A = np.random.rand(3, 3)
b = np.random.rand(3)
print "A:"
print A
print "b:"
print b
A, b = gauss_elim(A, b)
print "Gaussian Elimination:"
print A
print b
x = backsubst(A, b)
print "solution x:"
print x
print "solution using scipy:"
print solve(A, b)
