# -*- coding: utf-8 -*-

from __future__ import division, print_function
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# parameters of the system
gamma = 0.1
epsilon = 0.25
omega = 1.0
Omega = 2.0
Gamma = 1.5 # for chaos

# right hand side of the ode
def rhs(y, t):
    x = y[0]
    v = y[1]
    a = Gamma*np.cos(Omega * t) - gamma*v + omega**2*x - epsilon*x**3
    return np.array([v, a])

# time intervall to simulate
period_res = 100
T = 2 * 5000
period = 2*np.pi / Omega
step = period / period_res
t = np.arange(0, T, step)

# initial conditions
x0 = 1.0
v0 = 0.0
y0 = np.array([x0, v0])

# integration of the ode
start = time.time()
y = odeint(rhs, y0, t)
stop = time.time()
compute_time = stop - start
print("took %.2f seconds" % compute_time)
x = y[:, 0]
v = y[:, 1]

# plot the evolution of the solution $x$ and its derviative $\dot{x}$
first = 5000
first_time = first * step

plt.figure(1)
plt.subplot(2, 1, 1)
parameter_string = r"$\gamma = %.2f$" % gamma + r", $\epsilon = %.2f$" % epsilon + r", $\omega = %.2f$" % omega + ", $\Omega = %.2f$" % Omega + r", $\Gamma = %.2f$" % Gamma
plt.title("Time evolution for the first %.2f seconds of the Duffing oscillator, \nwith " % first_time + parameter_string)
plt.plot(t[:first], x[:first])
plt.ylabel("x [m]")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t[:first], v[:first])
plt.ylabel("v [m/s]")
plt.grid()
plt.xlabel("t [s]")

# plot the evolution of the system in phasespace
plt.figure(2)
plt.title("Phase space of the Duffing oscillator for the first %.2f seconds:\n" % first_time + r"with $x(0) = 1, \dot{x}(0) = 0$")
plt.plot(x[:first], v[:first])
plt.xlabel("x [m]")
plt.ylabel("v [m/s]")
plt.grid()

# plot the pointcare section the system
# the plot is accually a recurrence plot but dune to the Omega periodic externaml force
# we get approx. the same plot.
plt.figure(3)
x1 = x[::period_res] # external force has a 2pi/Omega period so period_res points
x2 = v[::period_res]
plt.plot(x1, x2, ".k", markersize=2)
plt.xlabel("x [m]")
plt.ylabel("v [m/s]")
plt.grid()
plt.title("Pointcare Section of the Duffing oscillator for %.2f seconds\n" % T + r"$\ddot{x} + \gamma \dot{x} - \omega^2 x + \epsilon x^3 = \Gamma \cos{(\Omega t)}$")

plt.show()
