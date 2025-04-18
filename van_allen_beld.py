import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.integrate import solve_ivp
import time

@numba.njit
def compute_magnetic_dipole_field(x, m):
    mu0 = 1.0
    r = np.linalg.norm(x)
    return mu0 / (4*np.pi) * (3 * np.dot(m, x) * x / r**5 - m / r**3)

@numba.njit
def rhs(t, y, q_over_m, m_earth):
    x, v = y[:3], y[3:]
    B = compute_magnetic_dipole_field(x, m_earth)
    a = q_over_m * np.cross(v, B)
    dy = np.empty(6)
    dy[:3] = v
    dy[3:] = a
    return dy

v_parallel_0 = 1e-8
v_orthogonal_0 = 1e-6
M = 10.0
m_earth = np.array([0.0, 0.0, M])
q_over_m = 1.0
L = 5.0
tmax = 1e7
x0 = np.array([L, 0.0, 0.0])
v0 = np.array([0.0, v_orthogonal_0, v_parallel_0])

start = time.perf_counter()
sol = solve_ivp(rhs, (0, tmax), np.hstack([x0, v0]), args=(q_over_m, m_earth), rtol=1e-6, atol=1e-6)
assert sol.success
print("time:", time.perf_counter() - start, "seconds")

plt.figure()
plt.plot(sol.y[1], sol.y[2], lw=0.05)
plt.xlabel("y (long)")
plt.ylabel("z (lat)")
plt.title("van allen beld")
plt.show()

