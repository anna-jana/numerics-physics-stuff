import numpy as np

def cholesky(A):
    G = np.zeros(A.shape)
    n = A.shape[0]
    for i in range(n):
        for j in range(i):
            G[i, j] = (A[i, j] - np.sum(G[i, :j] * G[j, :j])) / G[j, j]
        G[i, i] = np.sqrt(A[i, i] - np.sum(G[i, :i]**2))
    return G

A = np.array([[4,12,-16],
              [12,37,-43],
              [-16,-43,98]])
G = cholesky(A)
print("A:\n", A)
print("G:\n", G)
print("G*G':\n", G @ G.T)
print("||G*G' - A||:\n", np.linalg.norm(A - G @ G.T))
