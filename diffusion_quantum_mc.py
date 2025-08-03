# Diffusion Quantum Monte Carlo for a single particle
# https://arxiv.org/pdf/physics/9702023.pdf
#
# hbar = 1

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

k = 1.0

@njit
def V_fn(x):
    return 0.5*k*x**2

@njit
def dqmc(m, nwalkers0=50000, x0=0.0, Delta_tau=0.01, N_replicate_max=3, E_R_0=1.0, nsteps=1000):
    sigma = np.sqrt(Delta_tau / m)
    xs = [x0] * nwalkers0
    E_R = E_R_0

    for i in range(nsteps):
        print(f"{i} / {nsteps + 1}")
        new_xs = []
        for x in xs:
            x_new = x + sigma * np.random.randn()
            V_x = V_fn(x_new)
            W = np.exp(- (V_x - E_R) * Delta_tau)
            N_replicate = int(np.floor(W + np.random.rand()))
            N_replicate = min(N_replicate, N_replicate_max)
            for j in range(N_replicate):
                new_xs.append(x_new)
        N_prev = len(xs)
        N_next = len(new_xs)
        E_R = E_R + 1 / Delta_tau * (1 - N_next / N_prev)
        xs, new_xs = new_xs, xs

    return E_R, xs

m = 1
E_0, xs = dqmc(m)
nbins = 50
weights, bin_pos = np.histogram(xs, bins=nbins)
bin_width = bin_pos[1] - bin_pos[0]
c = weights / np.sqrt(np.sum(weights**2) * bin_width)
plt.plot(bin_pos[:-1] + bin_width/2, c, label="DQMC")
omega = np.sqrt(k/m)
s = (m*omega/np.pi)**(1/4)*np.exp(-m*omega*bin_pos**2/2)
plt.plot(bin_pos, s, label="analytic")
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$\psi_0$(x)")
plt.title(r"$E_0$ = 1/2 vs " + f"{E_0}")
plt.show()
