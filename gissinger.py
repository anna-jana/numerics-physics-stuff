import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rhs(t, u, mu, nu, gamma):
    x, y, z = u
    xdot = mu*x - y*z
    ydot = -nu*y + x*z
    zdot = gamma - z + x*y
    return xdot, ydot, zdot

mu = 0.119
nu = 0.1
gamma = 0.9
init = (3.0, 3.0, 3.0)
tmax = 500.0
sol = solve_ivp(rhs, (0, tmax), init, args=(mu, nu, gamma), dense_output=True)
ts = np.linspace(0, tmax, 1000)
plt.figure()
u = sol.sol(ts)
for i, l in enumerate(["x", "y", "z"]):
    plt.plot(ts, u[i, :], label=l)
plt.legend()
plt.xlabel("t")
plt.show()

