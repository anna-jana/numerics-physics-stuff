from __future__ import division

# Laplace equation:
# \Delta u = 0

# 1D:
# \frac{\partial^2 u}{\partial t^2} = 0

# Boundary conditions:
# u(0) = 0
# u(1) = 1

# Analytic solution:
# u(x) = x

# FDM:
# N \in \mathbb{N} is the number of grid points to use
# h is the step size
# \frac{u_{i - 1} - 2u_i + u_{i + 1}}{h^2} = 0 \forall i \in {1..N-2}
# u_{i - 1} - 2u_i + u_{i + 1} = 0 \forall i \in \{1..N-2}\}
# u_0 = 0
# u_{N - 1} = 1

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def one_D():
    # Construct and solve the linear system
    N = 100

    A = np.zeros((N, N))
    b = np.zeros(N)

    A[0, 0] = 1.0
    # b[0] = 0.0

    A[N - 1, N - 1] = 1.0
    b[N - 1] = 1.0

    for i in xrange(1, N - 1):
        A[i, i - 1] = 1.0
        A[i, i] = -2.0
        A[i, i + 1] = 1.0
        # b[i] = 0.0

    u = la.solve(A, b)

    # Plot the solution
    plt.plot(np.linspace(0, 1, u.size), u)
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title(r"$\Delta u = 0, u(0) = 0, u(1) = 1$ using FDM")
    plt.grid()
    plt.show()

# 2D:
# \Delta u = u_{xx} + u_{yy} = 0

# N is the number of columns (x coordinate)
# M is the number of columns (y coordinate)
# h_x is the step size in x direction
# h_y is the step size in y direction

# Boundary conditions:
u_00 = 0.0
u_01 = 1.0
u_11 = 2.0
u_10 = 3.0

grid_size = 30
N = grid_size # i
M = grid_size # j

num_vars = N*M

# n-th variable n = j*M + i
def cell_to_var(i, j):
    return j*M + i

A = np.zeros((num_vars, num_vars))
b = np.zeros(num_vars)

def linear_interpolate(x0, x1, y0, y1, x):
    rel = (x - x0)/(x1 - x0)
    return y0 + rel*(y1 - y0)

for i in xrange(N):
    # top
    j = 0
    n = cell_to_var(i, j)
    A[n, n] = 1.0
    b[n] = linear_interpolate(0, N - 1, u_00, u_10, i)
    # botton
    j = M - 1
    n = cell_to_var(i, j)
    A[n, n] = 1.0
    b[n] = linear_interpolate(0, N - 1, u_01, u_11, i)

for j in xrange(M):
    # left
    i = 0
    n = cell_to_var(i, j)
    A[n, n] = 1.0
    b[n] = linear_interpolate(0, M - 1, u_00, u_01, j)
    # right
    i = N - 1
    n = cell_to_var(i, j)
    A[n, n] = 1.0
    b[n] = linear_interpolate(0, M - 1, u_10, u_11, j)

# Finit Difference Stencil
# \frac{u_{i - 1, j} - 2u_{i,j} + u_{i + 1, j}}{h_x^2} + \frac{u_{i, j - 1} - 2u_{i, j} + u_{i, j + 1}}{h_y^2} = 0
# u_{i - 1, j} - 2u_{i,j} + u_{i + 1, j} + u_{i, j - 1} - 2u_{i, j} + u_{i, j + 1} = 0
# u_{i - 1, j} - 4u_{i,j} + u_{i + 1, j} + u_{i, j - 1} + u_{i, j + 1} = 0

for i in xrange(1, N - 1):
    for j in xrange(1, M - 1):
        n = cell_to_var(i, j)
        A[n, n] = -4.0
        A[n, cell_to_var(i - 1, j)] = 1.0
        A[n, cell_to_var(i + 1, j)] = 1.0
        A[n, cell_to_var(i, j - 1)] = 1.0
        A[n, cell_to_var(i, j + 1)] = 1.0
        # b[n] = 0.0

# solve the linear system
u = la.solve(A, b)
u = np.reshape(u, (M, N))

# plot the result

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, M)
plt.pcolormesh(x, y, u)
plt.title(r"$\Delta u = 0$ in 2D using FDM")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
