import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit
def rescale_to_temperature(v, T):
    E_kin = np.sum(v**2)
    N = v.shape[0]
    lamda = np.sqrt((N - 1) * 3 * T / E_kin)
    v *= lamda

@numba.njit
def minium_image_diff(x, i, j, L, diff):
    # minimum image convetion
    d = 0.0
    for n in range(3):
        diff[n] = x[i, n] - x[j, n]
        if abs(diff[n]) > L / 2:
            diff[n] -= np.round(diff[n] / L) * L
        d += diff[n]**2
    return np.sqrt(d)

@numba.njit
def compute_forces(x, out, L):
    N = x.shape[0]
    out[:, :] = 0.0
    diff = np.empty(3)
    for i in range(N):
        for j in range(i):
            dist = minium_image_diff(x, i, j, L, diff)
            d7 = dist**(-7)
            d14 = d7**2
            d8 = d7 / dist
            F = diff * 24 * (- 2 * d14 + d8)
            out[i, :] -= F
            out[j, :] += F

@numba.njit
def compute_potential(x, L):
    N = x.shape[0]
    E_pot = 0.0
    diff = np.empty(3)
    for i in range(N):
        for j in range(i):
            dist = minium_image_diff(x, i, j, L, diff)
            d6 = dist**(-6)
            d12 = d6**2
            E_pot += 4 * (d12 - d6)
    return E_pot

@numba.njit
def compute_kinetic(v):
    return 0.5 + np.sum(v**2)

@numba.njit
def verlet_step(x, v, F, F_new, dt):
    x += dt * v + 0.5 * dt**2 * F
    x %= L
    compute_forces(x, F_new, L)
    v += dt * 0.5 * (F + F_new)

M = 3
rho = 0.8
T = 1.0
dt = 0.004
rescale_every_n_steps = 10
init_time = 10.0
production_time = 20.0

# rho = L^3 / (4*M^3) = 1/4 * (L/M)^3
# L= M * (4rho)^(1/3)
N = 4 * M**3
L = M * np.cbrt(4*rho)
dx = L / M # periodic bc

# enforce vanishing total velocity/momentum
np.random.seed(100)
v = np.random.randn(N, 3) * np.sqrt(T)
total_momentum_per_particle = np.sum(v, axis=0) / N
v -= total_momentum_per_particle

nth = 0
x = np.empty_like(v)
for i in range(M):
    for j in range(M):
        for k in range(M):
            x[nth + 0, :] = [i,      j,       k]
            x[nth + 1, :] = [i + 0.5, j + 0.5, k]
            x[nth + 2, :] = [i + 0.5, j,       k + 0.5]
            x[nth + 3, :] = [i,       j + 0.5, k + 0.5]
            nth += 4
x *= dx

F = np.empty_like(v)
F_new = np.empty_like(v)

compute_forces(x, F, L)

init_steps = int(np.ceil(init_time / dt))
production_steps = int(np.ceil(production_time / dt))

def plot_points3d(x, color=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2], "o", color=color)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

if True:
    E_kins = []
    E_pots = []

    for i in range(init_steps + production_steps):
        print(f"{i + 1} / {init_steps + production_steps}")
        verlet_step(x, v, F, F_new, dt)
        F, F_new = F_new, F
        if i % rescale_every_n_steps == 0:
            rescale_to_temperature(v, T)
        if True: # i >= init_steps: # we are in production mode
            E_kins.append( 0.5*np.sum(v**2) )
            E_pots.append( compute_potential(x, L) )

    T = np.mean(E_kins) / ((N - 1)*3/2)
    T_err = np.std(E_kins) / ((N - 1)*3/2)
    E_pot = np.mean(E_pots) / N
    E_pot_err = np.std(E_pots) / N

