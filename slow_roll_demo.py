import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def pot(phi):
    return phi**2/2

def pot_deriv(phi):
    return phi

def calc_hubble(phi, phi_dot):
    rho = phi_dot**2/2 + pot(phi)
    H = np.sqrt(rho / 3)
    return H

def rhs(t, y):
    phi, phi_dot, a = y
    H = calc_hubble(phi, phi_dot)
    phi_dot_dot = - 3 * H * phi_dot - pot_deriv(phi)
    return (phi_dot, phi_dot_dot, H * a)

def solve(phi0, t_end=10.0):
    return solve_ivp(rhs, (0, t_end), (phi0, 0.0, 1.0), method="LSODA", rtol=1e-10)

fig1, ax1 = plt.subplots()
fig2, (ax2, ax3) = plt.subplots(2, 1)
for phi0 in np.linspace(0.001, 5.0, 10):
    sol = solve(phi0)
    ax1.plot(sol.y[0], sol.y[1], color="k")
    ax2.plot(sol.t, calc_hubble(sol.y[0], sol.y[1]))
    ax3.plot(sol.t, np.log(sol.y[2]))
ax1.set_xlabel(r"$\phi(t)$")
ax1.set_ylabel(r"$\dot{\phi}(t)$")
ax2.set_xlabel("time, t")
ax2.set_ylabel("Hubble parameter, H")
ax2.set_xscale("log")
ax3.set_xlabel("time, t")
ax3.set_ylabel("e-folds, log(a)")
ax3.set_yscale("log")
ax3.set_xscale("log")

plt.show()
