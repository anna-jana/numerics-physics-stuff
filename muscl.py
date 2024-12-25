# https://en.wikipedia.org/wiki/MUSCL_scheme

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

a = 1.0
@njit
def compute_flux(u):
    return a * u

tmax = 0.2
x = np.linspace(0, 1, 200)
dx = x[1] - x[0]
x0 = 0.1
x1 = 0.2
u_min = 0.0
u_max = 1.0
u0 = np.where((x < x1) & (x > x0), u_max, u_min)
u_analyic = np.where((x < x1 + a * tmax) & (x > x0 + a * tmax), u_max, u_min)

@njit
def linear_godunov_rhs(t, u):
    dudt = np.empty_like(u)
    dudt[0] = dudt[-1] = 0.0
    for i in range(1, u.shape[0] - 1):
        u_minus_half = (u[i - 1] + u[i]) / 2.0
        u_plus_half = (u[i] + u[i + 1]) / 2.0
        F_minus_half = compute_flux(u_minus_half)
        F_plus_half = compute_flux(u_plus_half)
        dudt[i] = - 1 / dx * (F_plus_half - F_minus_half)
    return dudt

@njit
def superbee_flux_limiter(u, i):
    if u[i + 1] == u[i]:
        return 2.0 if u[i] > u[i - 1] else 0.0
    else:
        r = (u[i] - u[i - 1]) / (u[i + 1] - u[i])
        return np.maximum(np.maximum(0.0, np.minimum(2*r, 1.0)), np.minimum(r, 2.0))

@njit
def muscl_kurganov_tadmor_rhs(t, u):
    dudt = np.empty_like(u)
    dudt[0:2] = dudt[-2:] = 0.0

    for i in range(2, u.shape[0] - 2):
        f = superbee_flux_limiter(u, i)
        f_plus = superbee_flux_limiter(u, i + 1)
        f_minus = superbee_flux_limiter(u, i - 1)

        u_L_plus_half = u[i] + 0.5 * f * (u[i + 1] - u[i])
        u_L_minus_half = u[i - 1] + 0.5 * f_minus * (u[i] - u[i - 1])

        u_R_plus_half = u[i + 1] - 0.5 * f_plus * (u[i + 2] - u[i + 1])
        u_R_minus_half = u[i] - 0.5 * f * (u[i + 1] - u[i])

        # since a is constant
        a_minus = a
        a_plus = a

        F_star_minus = (
            0.5 * (compute_flux(u_R_minus_half) + compute_flux(u_L_minus_half))
            - a_minus * (u_R_minus_half - u_L_minus_half)
        )
        F_star_plus = (
            0.5 * (compute_flux(u_R_plus_half) + compute_flux(u_L_plus_half))
            - a_plus * (u_R_plus_half - u_L_plus_half)
        )

        dudt[i] = - 1 / dx * (F_star_plus - F_star_minus)

    return dudt

sol_linear_godunov = solve_ivp(linear_godunov_rhs, (0, tmax), u0)
sol_muscl_kurganov_tadmor= solve_ivp(muscl_kurganov_tadmor_rhs, (0, tmax), u0)

plt.figure()
plt.plot(x, u0, label="initial")
plt.plot(x, u_analyic, label="analytic")
plt.plot(x, sol_linear_godunov.y[:, -1], ".-", label="linear godunov")
plt.plot(x, sol_muscl_kurganov_tadmor.y[:, -1], ".-", label="muscl kurganov tadmor")
plt.title("finite volume solution for advection equation")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.show()


