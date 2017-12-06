import numpy as np
import scipy.linalg as la


def qr_algorithm(A, eps=0.01, max_steps=1000):
    for i in range(max_steps):
        Q, R = la.qr(A)
        new_A = Q.T @ A @ Q
        diag_delta = la.norm(np.diag(A) - np.diag(new_A))
        if diag_delta < eps:
            return np.diag(A)
        A = new_A
    raise ValueError("failed to converge" + str(diag_delta))


A = np.array([[0,2,-1],[2,-1,1],[2,-1,3]])

print("eigenvalues of:\n", A)
print("scipy:", sorted(la.eigvals(A).real))
print("own qr algorithm:", sorted(qr_algorithm(A).real))
