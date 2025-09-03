import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1231324)

eps = 1e-4
required_flatness = 0.2
nbins = 10
E_min = 0
E_max = 1e3
Delta_E = (E_max - E_min) / nbins

histogram = np.zeros(nbins, dtype="int")
entropy = np.zeros(nbins)

def calc_E(x):
    return x**2

def get_bin(E):
    return int((E - E_min) / Delta_E)

x = np.random.randn()
n = get_bin(calc_E(x))

f = 1.0

while f > eps:
    new_x = x + np.random.randn()
    new_n = get_bin(calc_E(new_x))
    if new_n >= nbins: # upper limit for the energy
        continue
    prop = np.exp(entropy[n] - entropy[new_n])
    if np.random.rand() < prop:
        x = new_x
        n = new_n

    entropy[n] += f
    histogram[n] += 1

    H_mean = np.mean(histogram)
    flatness = np.max(np.abs((histogram - H_mean) / H_mean))

    if flatness < required_flatness:
        histogram[:] = 0
        f = f / 2.0
        print(f"reset histogram, decreasing to {f = }")

def noramlize(g):
    return g / (np.sum(g) * Delta_E)

plt.figure(layout="constrained")
energy = E_min + np.arange(nbins) * Delta_E + Delta_E / 2.0
density = noramlize(np.exp(entropy))
analytical_density = noramlize(energy**(-1/2))
plt.plot(energy, density, label="Wang-Landau")
plt.plot(energy, analytical_density, label="analytical")
plt.xlabel("energy E / a.u.")
plt.ylabel("normalized density of state / a.u.")
plt.legend()
plt.title("harmonic oscillator")
plt.show()
