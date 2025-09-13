import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

@njit
def H_to_t(H):
    return 1 / (2*H)

@njit
def t_to_H(t):
    return 1 / (2*t)

@njit
def rhs(t, u, N, M, G):
    H = H_to_t(t)
    phis = u[:N]
    phi_dots = u[N:]
    phi_dotdots = np.empty(N)
    for i in range(N):
        phi_dotdots[i] = - 3 * H * phi_dots[i] - M[i]**2 * phis[i]
        for j in range(N):
            phi_dotdots[i] += - G[i, j] * phis[i] * phis[j]**2
    out = np.empty(2*N)
    out[:N] = phi_dots
    out[N:] = phi_dotdots
    return out

@njit
def compute_number_density(u, N, M, G):
    phis = u[:N]
    phi_dots = u[N:]
    E_kin = 0.5 * phi_dots**2
    E_mass = M**2 * 0.5 * phis**2
    E_interaction = np.sum(G * phis[None, :]**2 * phis[:, None]**2, axis=0)
    E = E_kin + E_mass + E_interaction
    number_densities = E / M
    return np.sum(number_densities)

H0 = 100.0
H1 = 1e-3
t0 = H_to_t(H0)
t1 = H_to_t(H1)
np.random.seed(121231)
N = 20
phi0 = np.random.randn(N)
phi_dot0 = np.zeros(N)
M = np.random.randn(N)**2
G = np.random.randn(N, N)**2

sol = solve_ivp(rhs, (t0, t1), np.concatenate([phi0, phi_dot0]), args=(N, M, G), method="BDF", rtol=1e-5)

n = np.array([compute_number_density(sol.y[:, i], N, M, G) for i in range(sol.y.shape[1])])
H = t_to_H(sol.t)
s = H**(3/2)

plt.figure()
plt.loglog(H, n / s)
plt.xlabel("H")
plt.ylabel("total comoving number density")
plt.gca().invert_xaxis()
plt.show()
