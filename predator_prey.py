"""
Solve the predator prey or Lotka - Volterra equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def compute_predator_prey_rhs(z, t, alpha, beta, gamma, delta):
    x, y = z
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return dx_dt, dy_dt

alpha = 2/3
beta = 4/3
gamma = 1
delta = 1
x0 = 0.8
y0 = 0.9
t_span = 40
args = (alpha, beta, gamma, delta)
z0 = (x0, y0)
t = np.linspace(0, t_span, 400)

if __name__ == "__main__":
    x, y = odeint(compute_predator_prey_rhs, z0, t, args=args).T
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label="prey")
    plt.plot(t, y, label="predator")
    plt.xlabel("time t")
    plt.ylabel("popluation size")
    plt.title(r"Predator, Prey / Lotka - Volterra Model")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x, y)
    plt.title(r"$\alpha = %.2f, \beta = %.2f, \gamma = %.2f, \delta = %.2f$" % args)
    plt.xlabel("prey")
    plt.ylabel("predator")
    plt.tight_layout()
    plt.show()
