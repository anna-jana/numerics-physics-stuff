
import numpy as np

def power_eig(A, b0=None, max_steps=50, eps=1e-10):
    # set initial vector
    if b0 is None:
        b = np.random.rand(A.shape[0])
    else:
        b = b0
    # apply power method until we took too many steps or we converged
    for i in range(max_steps):
        new_b = A.dot(b)
        new_b /= np.linalg.norm(new_b)
        if np.linalg.norm(b - new_b) < eps:
            b = new_b # eigen vector
            break
        b = new_b
    else:
        raise ValueError("failed to converge")
    # compute eigen value
    # Ab = lb
    lb = A.dot(b)
    b_has_zero = b != 0
    eigen_val = np.mean(b[b_has_zero]/b[b_has_zero])
    return eigen_val, b

A = np.array([[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]], dtype="float")

val, vec = power_eig(A, b0=np.array([1,1,1]), eps=1e-5)

print("matrix:\n", A)
print("has eigen value", val, "with eigenvector", vec)




