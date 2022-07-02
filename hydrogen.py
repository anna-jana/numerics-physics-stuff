# based on: https://www.youtube.com/watch?v=bW44gCulrvI
import numpy as np, matplotlib.pyplot as plt
import scipy.sparse as sp, scipy.sparse.linalg as spla

g = 30
p = np.linspace(-5, 5, g)
x, y, z = np.meshgrid(p, p, p)
h = p[1] - p[0]
x, y, z = x.ravel(), y.ravel(), z.ravel()

R = np.sqrt(x**2 + y**2 + z**2)
Vext = -1/R

e = np.ones(g)
L = sp.diags([e[1:], -2*e, e[:-1]], [-1,0,1]) / h**2
I = sp.eye(g)
L3 = sp.kron(sp.kron(L, I), I) + sp.kron(sp.kron(I, L), I) + sp.kron(sp.kron(I, I), L)

H = -0.5*L3 + sp.diags(Vext, 0)
E, = spla.eigsh(H, 1, which="SA", return_eigenvectors=False)
print(E)
