from __future__ import division
import numpy as np
import scipy.linalg as la
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt

# Poissons equation:
# \Delta \phi = \rho(x)

# 2D:
# \Delta \phi = \phi_{xx} + \phi_{yy} = \rho/\epsilon_0

# N is the number of columns (x coordinate)
# M is the number of columns (y coordinate)
# hx is the step size in x direction
# hy is the step size in y direction

# charge density
def rho2(x, y, eps = 1e-2):
    charge_positions = [np.array([0.25, 0.5]), np.array([0.75, 0.5])]
    charge_mags = [-1, 1]
    vec = np.array([x, y])
    for p, q in zip(charge_positions, charge_mags):
        if np.linalg.norm(p - vec) < eps:
            return q
    return 0.0

def create_matrix(rho2, N, M, hx, hy):
        # N, M, <=> i, j
    num_vars = N*M

    if N is None:
        N = grid_size
    if M is None:
        M = grid_size

    A = np.zeros((num_vars, num_vars))
    b = np.zeros(num_vars)

# Boundary conditions:
# \partial\phi = 0
    def cell_to_var(i, j):
        # n-th variable n = j*M + i
        return j*M + i

    for i in xrange(N):
        # top
        j = 0
        n = cell_to_var(i, j)
        A[n, n] = 1.0
        b[n] = 0
        # botton
        j = M - 1
        n = cell_to_var(i, j)
        A[n, n] = 1.0
        b[n] = 0

    for j in xrange(M):
        # left
        i = 0
        n = cell_to_var(i, j)
        A[n, n] = 1.0
        b[n] = 0
        # right
        i = N - 1
        n = cell_to_var(i, j)
        A[n, n] = 1.0
        b[n] = 0

# Finit Difference Stencil
# \frac{u_{i - 1, j} - 2u_{i,j} + u_{i + 1, j}}{h_x^2} + \frac{u_{i, j - 1} - 2u_{i, j} + u_{i, j + 1}}{h_y^2} = \rho(i/N, j/M)*hx*hy/epsilon_0
# u_{i - 1, j} - 2u_{i,j} + u_{i + 1, j} + u_{i, j - 1} - 2u_{i, j} + u_{i, j + 1} = \rho(i/N, j/M)*hx*hy/epsilon_0
# u_{i - 1, j} - 4u_{i,j} + u_{i + 1, j} + u_{i, j - 1} + u_{i, j + 1} = \rho(i/N, j/M)*hx*hy/epsilon_0

    for i in xrange(1, N - 1):
        for j in xrange(1, M - 1):
            n = cell_to_var(i, j)
            A[n, n] = -4.0
            A[n, cell_to_var(i - 1, j)] = 1.0
            A[n, cell_to_var(i + 1, j)] = 1.0
            A[n, cell_to_var(i, j - 1)] = 1.0
            A[n, cell_to_var(i, j + 1)] = 1.0
            x = i / N
            y = j / M
            b[n] = rho2(x, y)*hx*hy/epsilon_0

    return A, b

grid_size = 40
hx = hy = 1.0
N = grid_size
M = grid_size

def direct_solver():

# creation of the matrix
    A, b = create_matrix(rho2, N, M, 1.0, 1.0)
# solve the linear system
    phi = la.solve(A, b)
    phi = np.reshape(phi, (M, N))
    return phi

def jakobi_iteration():
    A = np.random.rand(N, M)
    A[:,0] = A[:,-1] = A[0,:] = A[-1,:] = 0
    step = 1/M
    xs = np.linspace(step, 1 - step, M - 2)
    step = 1/N
    ys = np.linspace(step, 1 - step, N - 2)
    rhs = np.array([[rho2(x,y,eps=0.02)/epsilon_0 for x in xs] for y in ys])
    max_steps = 200
    for i in range(max_steps):
        north = A[:-2, 1:-1]
        south = A[2:, 1:-1]
        west = A[1:-1, :-2]
        east = A[1:-1, 2:]
        new_A = np.zeros(A.shape)
        new_A[1:-1, 1:-1] = (north + south + west + east - hx*hy*rhs)/4.0
        A = new_A
    return A

phi_direct = direct_solver()
phi_jakobi = jakobi_iteration()

# plot the result
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, M)
plt.subplot(2,1,1)
plt.pcolormesh(x, y, phi_jakobi)
plt.title(r"Poissons Equation $\Delta \phi = \rho/\epsilon_0$ in 2D using Jakobi Iteration")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.subplot(2,1,2)
plt.pcolormesh(x, y, phi_direct)
plt.title(r"Poissons Equation $\Delta \phi = \rho/\epsilon_0$ in 2D using FDM")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()



