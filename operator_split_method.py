import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq

def simulate_1d_schroedinger(V_fn, psi0_fn, m,
        L = 1.0, N = 1000, tspan = 10.0, dt = 0.01, save_interval = 5.0,):
    # grid and stepping
    dx = L / N # periodic boundary conditions
    x = np.arange(0, L, dx)
    nsteps = int(np.ceil(tspan / dt))
    k = 2*np.pi * fftfreq(N, dx)

    # initial condition
    psi0 = psi0_fn(x)
    psi0 = psi0 / (np.sum(psi0) * dx)
    psi_hat = fft(psi0)

    # propagator
    V = V_fn(x)
    kin_step = np.exp(- 1j * dt / 2 * k**2 / (2*m))
    pot_step = np.exp(- 1j * dt * V)

    # time stepping loop
    solutions = []
    save_step = int(np.floor(save_interval / dt))
    for i in range(nsteps + 1):
        if i % save_step == 0:
            solutions.append((i * dt, ifft(psi_hat)))
        psi_hat = kin_step * fft(pot_step * ifft(kin_step * psi_hat))

    print(solutions[0] is solutions[1])
    # plotting
    plt.figure(layout="constrained")
    plt.subplot(2,1,1)
    plt.plot(x, V)
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.subplot(2,1,2)
    for (t, psi) in solutions:
        plt.plot(x, np.abs(psi)**2, label=f"t = {t}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(t, x)|^2$")
    plt.show()

    return solutions


V0 = 100.0
sigma = 0.001
omega = 0.01
L = 1.0
x0 = 0.5 * L
n = 3
V_fn = lambda x: V0 * np.cos(2*np.pi*n*x/L)
psi0_fn = lambda x: np.exp(- (x - x0)**2 / (2*sigma)) * np.exp(- omega * 1j * (x - x0))
simulate_1d_schroedinger(V_fn, psi0_fn, 1.0, L=L)

