import numpy as np
from scipy import linalg as la

def lanczos_method(H, m, start=None):
    """
    Lanczos method of matrix H using m Krylov vectors. Uses a random vector a the first Krylov
    vector v_1 unless start is not None (then start is used).
    Returns a vector of the eigenvalues, a matrix of the eigen vectors (in the columns),
    a matrix of the Lanczos vectors (in the columns).
    """
    L = H.shape[0]
    assert 2 <= m <= L

    alpha = np.empty(m, dtype="complex") # diagonal of the resulting tridiagonal matrinx in krylov space
    beta = np.empty(m - 1, dtype="complex") # the subdiagonal

    if start is None:
        v_1 = np.random.rand(L) # inital vector (guess)
        v_1 /= np.linalg.norm(v_1)
    else:
        v_1 = start / la.norm(start)

    w_prime_1 = H @ v_1 # helper variable
    alpha[0] = np.dot(w_prime_1.conj(), v_1) # TODO: complex dot product correct?
    w_j_minus_1 = w_prime_1 - alpha[0] * v_1 # the previous w_j
    v_j_minus_1 = v_1 # the previous v_j
    lanczos_vectors = np.empty((L, m), dtype="complex")
    lanczos_vectors[:, 0] = v_1

    for j in range(1, m):
        beta[j - 1] = np.linalg.norm(w_j_minus_1)
        assert beta[j - 1] != 0.0
        v_j = w_j_minus_1 / beta[j - 1]
        w_prime_j = H @ v_j # helper variable
        alpha[j] = np.dot(w_prime_j.conj(), v_j)
        w_j = w_prime_j - alpha[j]*v_j - beta[j - 1]*v_j_minus_1
        lanczos_vectors[:, j] = v_j
        v_j_minus_1 = v_j
        w_j_minus_1 = w_j

    # compute eigenvalues of the tridiagonal matrix (in krylov space)
    #eigen_values, eigen_vectors = la.eigh_tridiagonal(alpha, beta)
    T = np.diag(alpha)
    I = np.arange(m - 1)
    T[I, I+1] = T[I+1, I] = beta
    eigen_values, eigen_vectors = la.eigh(T)
    return eigen_values, eigen_vectors, lanczos_vectors

if __name__ == "__main__":
    n = 20
    A = np.random.randn(n, n) + np.random.randn(n, n) * 1j
    A = (A + A.conj().T) / 2
    m = 5
    eig_vals, eig_vecs, _ = lanczos_method(A, m)
    def show(xs):
        return print(xs[np.argsort(-np.abs(xs))][:5])
    show(la.eigvals(A).real)
    show(eig_vals)

