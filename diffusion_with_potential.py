import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.integrate import simps
from numba import jit

@jit
def ftcs_step(psi, dt, dx, E_T, V):
    d2psidx2 = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / dx**2
    return psi + dt * (d2psidx2 / 2 - V * psi)

@jit
def crank_nicolson_step(psi, dt, dx, E_T, V):
    b = psi / dt + 1 / (4 * dx**2) * (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) - V * psi / 2
    next_psi = psi
    eps = 1e-5
    c = 1 / dt + 1 / (2*dx**2) + V / 2
    while True: # jacobi iteration
        new_next_psi = (b + (np.roll(next_psi, -1) + np.roll(next_psi, 1)) / (4 * dx**2)) / c
        change = np.linalg.norm(next_psi - new_next_psi)
        if change < eps:
            return new_next_psi
        next_psi = new_next_psi

def pseudo_spectral_step(psi, dt, dx, E_T, V):
    k = fftfreq(psi.size, dx) * 2*np.pi
    return np.real(np.exp(-V*dt/2) * ifft(np.exp(- k**2 / 2 * dt) * fft(np.exp(-V*dt/2) * psi)))

def solve(stepper, E_T, dt, L):
    x = np.linspace(0, L, N_x)
    dx = x[1] - x[0]
    print("dx =", dx)
    psi = np.zeros(N_x)
    psi[N_x // 2] = 1.0
    # psi = np.exp(- (x - L / 2)**2)
    psi_init = psi.copy()
    psi_norm = simps(psi**2, x)
    V = 1 / 2 * (L / np.pi * np.cos(np.pi * x / L))**2 - E_T
    # V = 0
    steps = int(tau_final / dt + 1)
    for i in range(steps):
        if i % 1000 == 0:
            print("step", i + 1, "of", steps)
            plt.clf()
            plt.plot(psi)
            plt.pause(0.01)
        psi = stepper(psi, dt, dx, E_T, V)
    print("initial norm squared:", psi_norm)
    print("final psi norm:", simps(psi**2, x))
    return x, psi, psi_init

def norm_sq(x, psi):
    return simps(psi**2, x)


if False:
    N_x = 200
    tau_final = 100
    L = 20
    dt = 0.001

    def f(E_T):
        print("***************************************************************")
        x, psi_final, psi_init = solve(pseudo_spectral_step, E_T, dt, L)
        norm_init = norm_sq(x, psi_init)
        norm_final = norm_sq(x, psi_final)
        return norm_init - norm_final

    from scipy.optimize import root

    ans = root(f, 1.0)
    assert ans.success
    E_T_star = ans.x[0]
    print(E_T_star)


if True:
    N_x = 200
    tau_final = 200
    L = 20
    E_T = 1.9391241920802921
    dt = 0.0001
    dt_ftcs = 0.0001
    x, psi_ftcs, psi_init = solve(ftcs_step, E_T, dt_ftcs, L)
    x, psi_cn, psi_init  = solve(crank_nicolson_step, E_T, dt, L)
    x, psi_ps, psi_init = solve(pseudo_spectral_step, E_T, dt, L)
    plt.plot(x, psi_ftcs, "-k", label="FTCS")
    plt.plot(x, psi_ps, "--r", label="Pseudo Spectral")
    plt.plot(x, psi_cn, ":b", label="Crank Nicolson")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\\psi$")
    plt.savefig("solution_b.pdf")
    plt.show()
