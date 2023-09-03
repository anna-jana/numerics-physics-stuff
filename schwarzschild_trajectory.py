import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rhs(_p, y, L, r_s):
    # e^nu = e^-lambda = 1 - r_s / r
    # e^(nu - lambda) = (1 - r_s / r)^2
    # d e^nu / d r = r_s / r^2 = e^nu * d nu / dr
    # d nu / dr = r_s / r^2 / e^nu = r_s / r^2 / (1 - r_s / r)
    # bc nu = - lambda => d lambda / dr = - d nu / dr
    phi, t, r, r_dot = y

    exp_nu = 1 - r_s / r
    exp_minus_lambda = exp_nu
    exp_nu_minus_lambda = exp_nu**2
    d_nu_dr = r_s / r**2 / exp_nu
    d_lambda_dr = - d_nu_dr

    phi_dot = L / r**2
    t_dot = 1 / exp_nu

    r_dot_dot = (
            - 0.5 * exp_nu_minus_lambda * d_nu_dr * t_dot**2
            - 0.5 * d_lambda_dr * r_dot**2
            + r * exp_minus_lambda * phi_dot**2
    )

    return phi_dot, t_dot, r_dot, r_dot_dot

def solve_and_plot(L, M, tspan=1e10, init_r=1e3, init_r_dot=0.0):
    r_s = 2 * M
    sol = solve_ivp(rhs, (0, tspan), (0.0, 0.0, init_r, init_r_dot),
            args=(L, r_s), method="BDF")
    phi, t, r, r_dot = sol.y
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    plt.plot(x, y, "-")

if __name__ == "__main__":
    solve_and_plot(0.1, 1)
    solve_and_plot(4.0, 1)
    # G M / r^2 = v^2 / r
    # v = sqrt(G M / r)
    # r omega = sqrt(G M / r)
    # r phi_dot = sqrt(G M / r)
    # r^(3/2) phi_dot = sqrt(G M)
    # phi_dot = sqrt(G M) / r^(3/2)
    # L = r^2 * phi_dot
    # L = r^2 sqrt(G M) / r^(3/2) = sqrt(G M / r)
    # r = 1e10
    # M = 1.0
    # L = np.sqrt(M / r)
    # solve_and_plot(L, M, init_r=r, tspan=1e20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


