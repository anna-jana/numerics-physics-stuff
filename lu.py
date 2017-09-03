from __future__ import print_function, division

import numpy as np
import scipy.linalg as la

def simple_lu(A):
    R = A.copy()
    L = np.eye(A.shape[0])
    for i in range(A.shape[0] - 1):
        factors = R[i + 1:, i] / R[i, i]
        R[i+1:, :] -= factors[:, None] * R[i, :]
        L[i+1:, i] = factors
    return L, R

A = np.array([[1.1, 2.1, 3.1],
              [2.2, 3.2, 1.2],
              [3.3, 1.3, 2.3]])

print("Simple LR without pivotisierung")
L, R = simple_lu(A)
print("A:\n", A)
print("L:\n", L)
print("R:\n", R)
print("L*R:\n", L.dot(R))


def lu(A):
    R = A.copy()
    L = np.eye(A.shape[0])
    p_rows = np.arange(A.shape[0])
    for i in range(A.shape[0] - 1):
        # pivot wahl
        pivot_row = i + np.argmax(np.abs(R[i:, i]))
        p_rows[i], p_rows[pivot_row] = p_rows[pivot_row], p_rows[i]
        R[i, :], R[pivot_row, :] = R[pivot_row, :], R[i, :].copy()
        # eliminate
        factors = R[i + 1:, i] / R[i, i]
        R[i + 1:, :] -= factors[:, None] * R[i, :]
        L[i + 1:, i] = factors
    P = np.zeros(A.shape)
    for i, p in enumerate(p_rows):
        P[i, p] = 1
    return P, L, R

print("Simple LR with pivotisierung")
P, L, R = lu(A)
print("A:\n", A)
print("L:\n", L)
print("R:\n", R)
print("P:\n", P)
print("L*R - P*A:\n", L.dot(R) - P.dot(A))

def backsubstitution(upper_triangular_matrix, rhs):
    sol = np.zeros(rhs.shape[0])
    for i in range(rhs.shape[0] - 1, -1, -1):
        sol[i] = (rhs[i] - np.sum(upper_triangular_matrix[i, i + 1:]*sol[i + 1:]))/upper_triangular_matrix[i, i]
    return sol

def forwardsubstituition(lower_triangular_matrix, rhs):
    sol = np.zeros(rhs.shape[0])
    for i in range(rhs.shape[0]):
        sol[i] = (rhs[i] - np.sum(lower_triangular_matrix[i, :i]*sol[:i]))/lower_triangular_matrix[i, i]
    return sol

def lu_solve(P, L, R, b):
    # Ax = b
    # PA = LR
    # P^-1*L*Rx = b
    # LRx = Pb
    # y = Rx
    # Ly = Pb
    # Rx = y
    y = forwardsubstituition(L, P.dot(b))
    x = backsubstitution(R, y)
    return x

b = np.array([1,2,3])
x = lu_solve(P, L, R, b)
print("b:\n", b)
print("x with Ax = b:\n", x)
print("Ax - b:\n", A.dot(x) - b)
