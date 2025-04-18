import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.integrate import solve_ivp
import time

@numba.njit
def rhs(t, y, M, g, mu0, m_pendulum, R, l, d, magnet_positions, m_magnet_vec):
    r, v = y[:3], y[3:]

    # forces from magnets (dipole - dipole interaction)
    n = r / R
    m_pendulum_vec = m_pendulum * n

    F_dipol = np.zeros(3)
    for r_magnet in magnet_positions:
        diff = r - r_magnet
        dist = np.linalg.norm(diff)
        diff_hat = diff / dist
        prefactor = 3 * mu0 / (4*np.pi*dist**4)
        term1 = np.cross(np.cross(diff_hat, m_magnet_vec), m_pendulum_vec)
        term2 = np.cross(np.cross(diff_hat, m_pendulum_vec), m_magnet_vec)
        term3 = - 2 * diff_hat * np.dot(m_magnet_vec, m_pendulum_vec)
        term4 = 5 * diff_hat * np.dot(np.cross(diff_hat, m_magnet_vec),
                                      np.cross(diff_hat, m_pendulum_vec))
        F_dipol += prefactor * (term1 + term2 + term3 + term4)

    # force from gravity
    F_gravity = np.array([0, 0, -g])
    F_real = F_dipol + F_gravity

    # constraint force which keeps the pendulum at fixed distance from origin
    F_constraint = - np.dot(F_real, n) * n
    F_total = F_real + F_constraint

    dy = np.empty(6)
    dy[:3] = v
    dy[3:] = F_total / M
    return dy

M = 1.0
g = 1.0
mu0 = 1.
m_pendulum = 1.0
R = 1.
l = 0.5
d = 0.0
h = np.sqrt(3) / 6 * l
magnet_positions = np.array([(0.0, 2 * h, d), (-l/2, - h, d), (+l/2, -h, d)])
m_magnet = 2.0
m_magnet_vec = m_magnet * np.array([0, 0, 1])
tspan = 200.0
v0 = np.zeros(3)
phi0 = 0
theta0 = np.pi + np.pi / 8
r0 = np.array([
    np.cos(phi0) * np.sin(theta0) * R,
    np.sin(phi0) * np.sin(theta0) * R,
    np.cos(theta0) * R
])

start = time.perf_counter()
sol = solve_ivp(rhs, (0, tspan), np.hstack([r0, v0]),
        args=(M, g, mu0, m_pendulum, R, l, d, magnet_positions, m_magnet_vec))
print("time:", time.perf_counter() - start, "seconds")
assert sol.success
x, y, z = sol.y[:3]

plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot([0, -l/2, +l/2], [h, -h, -h], "o")
plt.gca().set_aspect("equal")
plt.show()


