import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# constants
G = 1.0 # gravitational constant

# setup
M = 1.0 # mass of the central object
a = 2.0 # semi major axis
b = 1.0 # semi minor axis
# the orbit is in the x-y plane, semi major axis along x-axis, focal-point at origin

# initial conditions
M0 = 0.0 # M0 mean anomaly at epoch
t0 = 0.0 # epoch

# T: periode of the orbit
# circular orbit: T^2 = (4pi^2/GM) * r^3
# keplers 3rd law: T^2 / a^3 is the same for all orbits around the same central object
T = np.sqrt(4*np.pi / (G*M) * a**3)
n = 2*np.pi / T # average rate of sweep
e = np.sqrt(1 - b**2/a**2)  # eccentricity

# E: eccentric anaomly (position of the body along the elliptic orbit)
def goal(E, M): return E - e * np.sin(E) - M # keplers equation
def goal_prime(E, M): return 1 - e * np.cos(E)
def solve_for_E(M):
    sol = root_scalar(goal, args=(M,), x0=M, method="newton", fprime=goal_prime)
    assert sol.converged
    return sol.root
N = 100
# M = M0 + n*(t - t0)# mean anomaly
Ms = np.linspace(M0, M0 + 2*np.pi, N)
Es = np.array([solve_for_E(M) for M in Ms])

# cartesian coordianes of the object
xs = a * (np.cos(Es) - e)
ys = b * np.sin(Es)

plt.figure()
plt.plot(xs, ys, ".", color="blue", label="planet")
plt.plot([0], [0], "*", color="yellow", ms=10, label="star")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
