import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List
from numba.types import int64
import tqdm

@njit
def rescale_to_temperature(v, T):
    E_kin = np.sum(v**2)
    N = v.shape[0]
    lamda = np.sqrt((N - 1) * 3 * T / E_kin)
    v *= lamda

@njit
def minium_image_diff(x, i, j, L, diff):
    # minimum image convetion
    d = 0.0
    for n in range(3):
        diff[n] = x[i, n] - x[j, n]
        if abs(diff[n]) > L / 2:
            diff[n] -= np.round(diff[n] / L) * L
        d += diff[n]**2
    return np.sqrt(d)

@njit
def compute_force_between(x, n, m, L, diff, out):
    dist = minium_image_diff(x, n, m, L, diff)
    d7 = dist**(-7)
    d14 = d7**2
    d8 = d7 / dist
    F = diff * 24 * (- 2 * d14 + d8)
    out[n, :] -= F
    out[m, :] += F

@njit
def box_iter(x, L, n_box_per_side):
    N = x.shape[0]
    L_box = L / n_box_per_side

    boxes = [[[List.empty_list(int64)
               for _ in range(n_box_per_side)]
              for _ in range(n_box_per_side)]
             for _ in range(n_box_per_side)]

    box_ids = np.empty((N, 3), dtype="int")

    for n in range(N):
        i = int(x[n, 0] / L_box)
        j = int(x[n, 1] / L_box)
        k = int(x[n, 2] / L_box)
        boxes[i][j][k].append(n)
        box_ids[n, 0] = i
        box_ids[n, 1] = j
        box_ids[n, 2] = k

    for n in range(N):
        i = box_ids[i, 0]
        j = box_ids[i, 1]
        k = box_ids[i, 2]
        for i_off in i-1, i, i+1:
            i_off %= n_box_per_side
            for j_off in j-1, j, j+1:
                j_off %= n_box_per_side
                for k_off in k-1, k, k+1:
                    k_off %= n_box_per_side
                    for m in boxes[i_off][j_off][k_off]:
                        if n < m:
                            yield n, m

@njit
def compute_forces(x, out, L):
    out[:, :] = 0.0
    diff = np.empty(3)

    sigma = 2.0 # in code units
    n_box_per_side = int(L / sigma)

    if n_box_per_side > 3:
        for n, m in box_iter(x, L, n_box_per_side):
            compute_force_between(x, n, m, L, diff, out)
    else:
        # no box code
        N = x.shape[0]
        for n in range(N):
            for m in range(n):
                compute_force_between(x, n, m, L, diff, out)

@njit
def compute_potential_between(x, n, m, L, diff):
    dist = minium_image_diff(x, n, m, L, diff)
    d6 = dist**(-6)
    d12 = d6**2
    return 4 * (d12 - d6)

@njit
def compute_potential(x, L):
    diff = np.empty(3)
    E_pot = 0.0

    sigma = 2.0 # in code units
    n_box_per_side = int(L / sigma)

    if n_box_per_side > 3:
        for n, m in box_iter(x, L, n_box_per_side):
            E_pot += compute_potential_between(x, n, m, L, diff)
    else:
        # no box code
        N = x.shape[0]
        for n in range(N):
            for m in range(n):
                E_pot += compute_potential_between(x, n, m, L, diff)

    return E_pot

@njit
def compute_kinetic(v):
    return 0.5 + np.sum(v**2)

@njit
def verlet_step(x, v, F, F_new, dt):
    # velocity verlet method
    x += dt * v + 0.5 * dt**2 * F
    # move particles into box L^3
    x %= L
    compute_forces(x, F_new, L)
    v += dt * 0.5 * (F + F_new)

def init_grid(M, N, L):
    nth = 0
    x = np.empty((N, 3))
    for i in range(M):
        for j in range(M):
            for k in range(M):
                x[nth + 0, :] = [i,       j,       k      ]
                x[nth + 1, :] = [i + 0.5, j + 0.5, k      ]
                x[nth + 2, :] = [i + 0.5, j,       k + 0.5]
                x[nth + 3, :] = [i,       j + 0.5, k + 0.5]
                nth += 4
    dx = L / M # periodic bc
    x *= dx
    return x

def init_velocity_maxwell(N, T):
    v = np.random.randn(N, 3) * np.sqrt(T)
    # enforce vanishing total velocity/momentum
    total_momentum_per_particle = np.sum(v, axis=0) / N
    v -= total_momentum_per_particle
    return v

if __name__ == "__main__":
    np.random.seed(100)
    M = 3
    rho = 0.8
    dt = 0.004
    rescale_every_n_steps = 10
    init_time = 5.0
    production_time = 5.0

    N = 4 * M**3
    L = M * np.cbrt(4*rho)

    Ts = np.linspace(4.5, 6.0, 10)
    init_steps = int(np.ceil(init_time / dt))
    production_steps = int(np.ceil(production_time / dt))
    data = []

    for T in Ts:
        x = init_grid(M, N, L)
        v = init_velocity_maxwell(N, T)

        F = np.empty_like(v)
        F_new = np.empty_like(v)
        compute_forces(x, F, L)

        E_kins = np.empty(production_steps)
        E_pots = np.empty(production_steps)

        for i in tqdm.tqdm(range(init_steps + production_steps)):
            verlet_step(x, v, F, F_new, dt)
            F, F_new = F_new, F
            if i % rescale_every_n_steps == 0:
                rescale_to_temperature(v, T)
            if i >= init_steps: # we are in production mode
                E_kins[i - init_steps] = 0.5*np.sum(v**2)
                E_pots[i - init_steps] = compute_potential(x, L)

        E = E_kins + E_pots
        data.append(E)

    # post processing: compute heat capacity at V = const
    E_mean = np.array(list(map(np.mean, data)))
    E_err = np.array(list(map(np.std, data)))
    Delta_T = np.diff(Ts)
    c_v = np.diff(E_mean) / Delta_T / N
    c_v_err = np.sqrt(E_err[:-1]**2 + E_err[1:]**2) / Delta_T / N
    dof = 3
    E_ideal = dof/2*N*Ts
    c_v_ideal = (3 + dof) / 2

    fig, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
    plt.subplot(2,1,1)
    axs[0].errorbar(Ts, E_mean, yerr=E_err, fmt="o", label="Lennard Jones simulation")
    axs[0].plot(Ts, E_ideal, ls="--", label="ideal gas")
    axs[0].set_ylabel("internal energy E / a.u.")
    axs[0].legend()
    axs[1].errorbar((Ts[:-1] + Ts[1:]) / 2.0,  c_v, yerr=c_v_err, fmt="o", label="Lennard Jones simulation")
    axs[1].axhline(c_v_ideal, ls="--", color="tab:orange", label="ideal gas")
    axs[1].set_xlabel("temperature T / a.u.")
    axs[1].set_ylabel("heat capacity at const. vol. c_v * k_B")
    #axs[1].legend()
    plt.show()
