import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg

n = 10
L = 1.0
phi_bc = 10.0
f0 = 1.0 / (L / (n - 1))**2

indicies = np.arange(n**2).reshape(n,n)
triangles = np.array((
    [(indicies[i, j], indicies[i + 1, j + 1], indicies[i, j + 1]) for i in range(n - 1) for j in range(n - 1)] +
    [(indicies[i, j], indicies[i + 1, j], indicies[i + 1, j + 1]) for i in range(n - 1) for j in range(n - 1)]
))

X = Y = np.linspace(0, L, n)
xx, yy = np.meshgrid(X, Y)
x = xx.reshape(-1)
y = yy.reshape(-1)

def dist(tri):
    tri_x = x[tri]
    tri_y = y[tri]
    return tri_x[:, None] - tri_x[None, :], tri_y[:, None] - tri_y[None, :]

is_boundary = np.zeros(indicies.shape, dtype=bool)
is_boundary[0,:] = is_boundary[:,0] = is_boundary[-1,:] = is_boundary[:,-1] = True
is_boundary = is_boundary.reshape(-1)

phi0 = np.zeros(n*n)
phi0[is_boundary] = phi_bc

f = np.zeros((n, n))
f[f.shape[0] // 2, f.shape[1] // 2] = f0
f = f.reshape(-1)

r = np.zeros(n*n)
f_to_r_mat = (np.eye(3) + np.ones((3, 3))) / 24
for tri in triangles:
    xx, yy = dist(tri)
    two_A = xx[1, 0] * yy[2, 0] - xx[2, 0] * yy[1, 0]
    r[tri] += two_A * f_to_r_mat @ f[tri]

r[is_boundary] = phi_bc

def local_stiffness_matrix(tri):
    xx, yy = dist(tri)
    b = np.array([[yy[1, 2], yy[2, 0], yy[0, 1]]])
    c = np.array([[xx[2, 1], xx[0, 2], xx[1, 0]]])
    return b.T @ b + c.T @ c

local_stiffness_matricies = list(map(local_stiffness_matrix, triangles))

def apply_stiffness_matrix(phi):
    ans = np.zeros(len(phi))
    ans[is_boundary] = phi[is_boundary]
    for tri, k in zip(triangles, local_stiffness_matricies):
        ans[tri] += np.where(is_boundary[tri], 0.0, k @ phi[tri])
    return ans

K = LinearOperator((len(x), len(x)), apply_stiffness_matrix)

phi, status = cg(K, r, x0=phi0, tol=1e-10)
assert status == 0

plt.figure()
plt.tripcolor(x, y, phi, triangles=triangles, shading="gouraud")
for i, j, k in triangles:
    plt.plot([x[i], x[j], x[k], x[i]], [y[i], y[j], y[k], y[i]], color="red")
plt.xlabel("x / a.u")
plt.ylabel("y / a.u")
plt.colorbar(label=r"field $\phi$ in a.u")
plt.title("fem solution to poisson eq.")
plt.show()
