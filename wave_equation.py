import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft, ifft
from scipy.integrate import solve_ivp

c = 1.0

L = 2*np.pi
N = 100
xs = np.linspace(0, L, N+1)[:-1]
dx = xs[1] - xs[0]

dt = 1e-2
tspan = 1e2
nsteps = int(np.ceil(tspan / dt))
t_end = dt * nsteps

A = 1.0
sigma = 1.0
phi0_fn = lambda x: A*np.exp(-(x - L/2)**2 / sigma**2)
phi0 = phi0_fn(xs)
phi_dot0_fn = lambda x: 0.0*x
phi_dot0 = phi_dot0_fn(xs)

C = c**2 * dt**2 / dx**2

f_R = lambda x: 0.5 * (phi_dot0_fn(x) / c + phi0_fn(x))
f_L = lambda x: phi0_fn(x) - f_R(x)
phi_fn = lambda t, x: f_L(np.mod(x - c*t, L)) + f_R(np.mod(x + c*t, L))
phi_end = phi_fn(t_end, xs)

def compute_F(phi):
    return c**2 * (np.roll(phi, 1) - 2*phi + np.roll(phi, -1)) / dx**2

def verlet_step(phi, phi_dot, F):
    phi_new = phi + dt * phi_dot + 0.5 * dt**2 * F
    F_new = compute_F(phi_new)
    phi_dot_new = phi_dot + dt * 0.5 * (F + F_new)
    return phi_new, phi_dot_new, F_new

def ctcs_step(prev_phi, phi):
    return phi, 2*phi - prev_phi + dt**2*c**2*compute_F(phi)

k = 2*np.pi*fftfreq(N, dx)

def pseudo_spectral_rhs(_, y):
    phi_fft = y[:N]
    phi_dot_fft = y[N:]
    return np.hstack([phi_dot_fft, - k**2 * phi_fft])

phi = phi0
phi_dot = phi_dot0
F = compute_F(phi)
for i in range(nsteps):
    phi, phi_dot, F = verlet_step(phi, phi_dot, F)
phi_verlet = phi

prev_phi = phi0
phi = phi0 + dt * phi_dot0
for i in range(nsteps - 1):
    prev_phi, phi = ctcs_step(prev_phi, phi)
phi_ctcs = phi

sol = solve_ivp(pseudo_spectral_rhs, (0.0, nsteps * dt),
        np.hstack([fft(phi0), fft(phi_dot0)]))
phi_pseudo_spectral = ifft(sol.y[:, -1][:N])

plt.figure()
plt.plot(xs, phi_end, label="analytical", color="k")
plt.plot(xs, phi_verlet, "x", label="verlet")
plt.plot(xs, phi_ctcs, "*", label="ctcs")
plt.plot(xs, phi_pseudo_spectral, ".", label="pseudo spectral")
plt.xlabel("x")
plt.ylabel(r"$\phi(t, x)$")
plt.legend(loc=2)
plt.title(f"wave eq. t = {t_end:.2e}")
plt.show()
