import numpy as np

np.random.seed(42)

def cg(A, b, x0=None, kmax=None, eps=1e-10):
    x = np.random.randn(b.shape[0]) if x0 is None else x0
    r = b - A @ x
    p = r
    k = 0
    r_sq_old = r.T @ r
    while True:
        alpha = r.T @ r / (p.T @ A @ p)
        x = x + alpha * p
        r = r - alpha * A @ p
        r_sq_new = r.T @ r
        if r_sq_new < eps: return x
        beta = r_sq_new / r_sq_old
        p = r + beta * p
        k += 1
        if kmax is not None and k > kmax: raise ValueError("not many steps")
        r_sq_old = r_sq_new

A = np.array([[4,1],[1,3]])
b = np.array([1,2])
x = cg(A, b)
print("A:", A, "b:", b, "x:", x, sep="\n")

