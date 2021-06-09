# implement  FTTD for maxwell equation in matter (1D)
# based on:  https://www.youtube.com/watch?v=OjbfxnfCWRQ

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.fft as fft


# physical constants
epsilon0 = 1 # electric vacuum constant
mu0 = 1 # magnetic vacuum constant
omega = 10 # frequency of polarization oscillators
rho = 1 # mass density of polarization oscillators
kappa = 5 # coupling strength of electric field to polarization oscillators
c = 1 / np.sqrt(epsilon0 * mu0) # speed of light

# discretisation
h = 0.1 # grid spacing
L = 100 * h # length of the domain
dt = 0.03 # time step
tspan = 1000 * dt # time to simulate
z = np.arange(0, L, h) # postion of the grid (B_y)
n = z.size # number of grid points
num_steps = int(tspan / dt) # number of time steps

# initial field configurations
E_x = np.exp(-(z - L/2)**2) # electric field
B_y = np.zeros(n) # magnetic field
P = np.zeros(n) # polarization excitation also in x direction
Y = np.zeros(n) # velocity of polarization oscillators

# finite difference operators (periodic boundary condtions)
e = np.ones(n)
D_f = sp.diags([1, -e, e], [-n + 1, 0, 1], (n, n)) / h # forward
D_b = sp.diags([-e, e, -1], [-1, 0, n - 1], (n, n)) / h # backward

# record history for analysis of the dispersion relation
E_x_s = []

for i in range(num_steps):
    # propagate only hafe timestep to have E and B at the same time point
    #dBydt = - D_f @ E_x
    #B_y += dBydt * dt / 2
    #dExdt = - c**2 * D_b @ B_y
    #E_x += dExdt * dt
    #dBydt = - D_f @ E_x
    #B_y += dBydt * dt / 2
    # update (finite difference forward time) (FDFT method)
    dYdt = -omega**2*P - kappa*1/rho*E_x
    Y += dYdt*dt
    dBydt = - D_f @ E_x
    B_y += dBydt * dt
    dExdt = - c**2 * D_b @ B_y + 1/epsilon0*Y
    P += Y*dt
    E_x += dExdt *dt

    E_x_s.append(E_x.copy()) # record history

    # display results
    if i % 10 == 0:
        plt.clf()
        plt.plot(z - h/2, E_x, ls="--", label="E_x") # offset of E grid (Yee grid)
        plt.plot(z - h/2, P, ls="--", label="P")
        plt.plot(z, B_y, label="B_y")
        plt.plot(z, Y, label="Y")
        plt.legend(loc=1)
        plt.xlabel("z")
        plt.ylabel("fields")
        plt.pause(0.001)


# post processing (analysis of dispersion relation)
plt.clf()
k_space = fft.fftn(E_x_s)
plt.pcolormesh(np.log10(np.abs(k_space[:100, :])), cmap="viridis")
plt.xlabel("k")
plt.ylabel(r"$\omega$")
plt.show()
