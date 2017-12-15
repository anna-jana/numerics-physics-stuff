from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x == 0:
        return 1
    return x/abs(x)

def qr(A):
    A = A.copy()
    m,n = A.shape
    U = np.zeros((m,n))
    for k in range(n):
        # berechnung der k-ten householder matrix
        A_k_max = np.max(np.abs(A[k:, k]))     # k: das geht von k bis m
        alpha = 0
        for i in range(k, m):
            U[i, k] = A[i, k]/A_k_max
            alpha += np.abs(U[i, k])**2
        alpha = np.sqrt(alpha)
        beta_k = 1/(alpha*(alpha + np.abs(U[k,k])))
        U[k,k] += sign(A[k,k]) * alpha
        # multiplikation der k-ten householder matrix
        A[k,k] = -sign(A[k,k]) * alpha * A_k_max
        for i in range(k + 1, m):
            A[i,k] = 0
        for j in range(k + 1, n):
            s = beta_k * np.sum(U[k:, k] * A[k:, j])
            for i in range(k, m):
                A[i,j] -= s*U[i,k]
    # berechnung von Q
    Q = np.eye(m)
    for k in range(n):
        u_k = U[:, k]
        H_k = np.eye(m) - 2/(u_k @ u_k) * (u_k[:, None] @ u_k[None, :])
        Q = Q @ H_k
    return Q, A


def linear_regression(x, y):
    A = np.vstack([x, np.repeat(1, x.size)]).T
    Q, R = qr(A)
    R_hat = R[:2, :2]
    Q_T_y = Q.T @ y
    c = Q_T_y[:2]
    parameter = np.linalg.solve(R_hat, c)
    return parameter


if __name__ == "__main__":
    A = np.array([[0.0, -4, 2],
                  [6, -3, -2],
                  [8, 1, -1]])
    Q, R = qr(A)
    print("A:\n", A)
    print("Q:\n", Q)
    print("R:\n", R)
    print("Q' @ Q:\n", Q.T @ Q)
    print("Q @ R - A:\n", Q @ R - A)

    n = 30
    x = np.linspace(0, 10, n)
    y = 2*x + 3 + np.random.randn(n)*3
    parameter = linear_regression(x, y)
    fit = parameter[0]*x + parameter[1]
    plt.plot(x, y, "xk", label="Data")
    plt.plot(x, fit, "-k", label="Linear Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

