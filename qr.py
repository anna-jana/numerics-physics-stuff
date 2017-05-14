import numpy as np

def gram_schmidt(ws):
    vs = ws.copy()
    num_vectors = ws.shape[1]
    # orthogonize
    for i in range(num_vectors):
        for j in range(i):
            vs[:, i] -= vs[:, j].dot(ws[:, i])/(vs[:, j].dot(vs[:, j]))*vs[:, j]
    # normalize
    for i in range(num_vectors):
        vs[:, i] /= np.linalg.norm(vs[:, i])
    return vs

def qr(A):
    # A = S_id_B
    Q = gram_schmidt(A) # = S_id_B'
    # dabei ist die basiswechselmatrix von B nach B' eine obere dreiecksmatrix und damit auch die
    # von B' nach B
    #             Q          R
    #             |          |
    #             V          V
    # Also A = S_id_B' * B'_id_B
    #          ~~~~~~
    #          Gramschmidt
    # => A = Q*R => Q^-1*A = R => (da Q orthogonal) Q'*A = R
    R = Q.transpose().dot(A)
    return Q, R

A =  np.array([[1,2.],[3,4]])
Q, R = qr(A)

print "A:\n", A, "\nQ:\n", Q, "\nR:\n", R, "\nQ*R:\n", Q.dot(R)
