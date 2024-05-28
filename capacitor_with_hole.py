import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# physical parameters
Lx = 10.0
Ly = 10.0
d_electrodes = 2.0
d_hole = 2.0
V = 1.0

# numerical parameters
Nx = 100
Ny = 100
eps = 1e-15

# geometry computations
x_anode = Lx / 2 - d_electrodes / 2
x_katode = Lx / 2 + d_electrodes / 2
dx = Lx / (Nx - 1)
i_anode = int(x_anode / dx)
i_katode = int(x_katode / dx)

y_hole_start = Ly / 2 - d_hole / 2
y_hole_end = Ly / 2 + d_hole / 2
dy = Ly / (Ny - 1)
j_hole_start = int(y_hole_start / dy)
j_hole_end = int(y_hole_end / dy)

# gauss-seidel method
@njit
def compute(phi):
    while True:
        mse = 0.0
        for i in range(1, Nx - 1):
            for j in range(Ny):
                if i in (i_anode, i_katode) and not (j_hole_start <= j <= j_hole_end):
                    continue
                new = (phi[i, (j - 1) % Ny] + phi[i, (j + 1) % Ny] +
                        phi[i - 1, j] + phi[i + 1, j]) / 4
                mse += (new - phi[i, j])**2
                phi[i, j] = new
        mse /= (Nx * Ny)
        if mse < eps:
            break

# initial guess for field
phi = np.zeros((Nx, Ny))
phi[i_anode, :j_hole_start] = +V
phi[i_anode, j_hole_end:] = +V
phi[i_katode, :j_hole_start] = -V
phi[i_katode, j_hole_end:] = -V
# run solver
compute(phi)

# plot
plt.figure()
xs = np.linspace(0, Lx, Nx)
ys = np.linspace(0, Ly, Ny)
plt.contourf(xs, ys, phi)
plt.colorbar(label=r"Potential $\phi$ / a.u.")
plt.plot([0, y_hole_start], [x_anode, x_anode],  "k", lw=2.0)
plt.plot([y_hole_end, Ly], [x_anode, x_anode], "k", lw=2.0)
plt.plot([0, y_hole_start], [x_katode, x_katode],  "k", lw=2.0)
plt.plot([y_hole_end, Ly], [x_katode, x_katode], "k", lw=2.0)
plt.xlabel("x / a.u.")
plt.ylabel("y / a.u.")
plt.title("Capacitor with hole")
plt.show()
