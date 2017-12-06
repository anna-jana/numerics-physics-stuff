from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # return np.sin(x) / x
    return np.exp(-0.001*x**2) + np.sin(x) / x

steps = 100
jump_radius = 10.0
init_radius = 20.0
x = np.random.uniform(-init_radius, init_radius)
energy = f(x)

x_hist = [x]
f_hist = [energy]

for i in range(1, steps + 1):
    # delta_x = np.random.uniform(-jump_radius, jump_radius)
    delta_x = np.random.randn()*jump_radius
    new_x = x + delta_x
    new_energy = f(new_x)
    delta_energy = energy - new_energy
    temperature = np.exp(- steps / i**2)
    jump_probability = np.exp(- delta_energy / temperature)
    if np.random.rand() <= jump_probability:
        x_hist.append(new_x)
        f_hist.append(new_energy)
        x = new_x
        energy = new_energy

xs = np.linspace(2*min(x_hist), 2*max(x_hist), 300)
plt.plot(xs, f(xs), label="f(x)")
plt.plot(x_hist, f_hist, "k+", label="steps")
plt.plot(x_hist[-1], f_hist[-1], "ro", label="final position")
plt.plot(x_hist[0], f_hist[0], "go", label="initial position")
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Simulated Annealing")
plt.show()

