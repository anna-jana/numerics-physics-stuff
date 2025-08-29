import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

k = 1.0
t_span = 100.0
noise = 0.1
dt = 1e-1
x0 = 0.0
v0 = 0.0
x_goal = np.pi
x_goal_deriv = 0.0
K_P = 3.0
K_I = 0.2
K_D = 1.0

x = x0
v = v0
nsteps = int(t_span / dt) + 1
xs = []
I = 0.0

for i in range(nsteps):
    # controler
    P = K_P * (x_goal - x)
    D = K_D * (x_goal_deriv - v)
    I += K_I * dt * (x_goal - x)
    F = P + I + D

    # advance ode
    rhs = lambda t, y: (y[1], -k*np.sin(y[0]) + F)
    sol = solve_ivp(rhs, (0, dt), (x, v))
    assert sol.success
    x, v = sol.y[:, -1]
    xs.append(x)
    # kick
    v += dt * np.random.randn() * noise

plt.figure(layout="constrained")
plt.plot(np.arange(nsteps) * dt, xs, label="actual")
plt.axhline(x_goal, ls="--", label="goal")
plt.plot([0.0], [x0], "o", label="initial")
plt.xlabel("t / a.u")
plt.ylabel("angle / radians")
plt.legend()
plt.show()
