
# coding: utf-8

import numpy as np

def jacobi_method(A, b, eps=1e-5, max_steps=100, debug=False):
    """
    Solves a linear system Ax = b for x using the iterative jacobi method,
    to a requested error eps with at most max_steps steps.
    Turn on debug prints using debug=True.
    Returns the solution x. Raises a ValueError if the iteration doesn't converge.
    """

    cols = np.arange(A.shape[0])[None,:].repeat(A.shape[1], 0)
    rows = np.arange(A.shape[1])[:,None].repeat(A.shape[0], 1)

    L = (rows > cols) * A
    U = (rows < cols) * A
    D = np.diag(A)

    x = np.zeros_like(b)

    for i in range(max_steps):
        x =  (1.0 / D) * (b - (L + U) @ x)
        err = np.linalg.norm(A @ x - b)
        if debug:
            print("error:", err, "\nx:\n", x)
        if err <= eps:
            return x

    raise ValueError("Failed to converge.\neps = {}, max_steps = {}\nA:\n{}\nb:\n{}\nlast x:\n{}".format(eps, max_steps, A, b, x))

if __name__ == "__main__":

    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0.0, 3., -1., 8.]])
    b = np.array([6., 25., -11., 15.])


    jacobi_x = jacobi_method(A, b, eps=0)
    numpy_x = np.linalg.solve(A, b)

    print("A:\n", A, "\nb:\n", b, "\nx using jacobi:\n", jacobi_x, "\nerror:", np.linalg.norm(jacobi_x - numpy_x))
