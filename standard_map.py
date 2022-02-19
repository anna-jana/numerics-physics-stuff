import numpy as np, matplotlib.pyplot as plt
from numba import jit

@jit
def standard_map(x, p, K):
    new_p = (p + K*np.sin(x)) % (2*np.pi)
    return (x + new_p) % (2*np.pi), new_p

@jit
def std_map_phase_space(K, grid_steps=20, orbit_steps=100):
    orbits = np.empty((grid_steps**2, orbit_steps, 2))
    i = 0
    for x in np.linspace(0, 2*np.pi, grid_steps):
        for p in np.linspace(0, 2*np.pi, grid_steps):
            for j in range(orbit_steps):
                x, p = standard_map(x, p, K)
                orbits[i, j, 0] = x
                orbits[i, j, 1] = p
            i += 1
    return orbits

def plot_orbits(orbits):
    for i in range(orbits.shape[0]):
        plt.plot(orbits[i, :, 0], orbits[i, :, 1], "o", ms=0.5)
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title(f"K = {K:.2}")

for i, K in enumerate([0.5, 0.971635, 2.0]):
    orbits = std_map_phase_space(K)
    plt.subplot(1, 3, i + 1)
    plot_orbits(orbits)
plt.suptitle("Standard Map")
plt.tight_layout()
plt.show()
