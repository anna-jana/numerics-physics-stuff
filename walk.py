import numpy as np
import matplotlib.pyplot as plt

steps = 500
do_walks = 500
end_points = np.empty(do_walks)
dx = 1

for i in range(do_walks):
    # the result is independent of the distribution of the induvidual steps
    # (central limit theorem)
    # deltas = np.random.uniform(-1,1,steps) * dx
    # deltas = (1 - 2 * np.random.randint(2, size=steps)) * dx
    deltas = np.random.normal(loc=0, scale=dx, size=steps)
    dist = np.cumsum(deltas)
    end_points[i] = dist[-1]

plt.subplot(2,1,1)
plt.hist(end_points, density=True, bins=31, label="MC Simulation")
std = steps**(1/2) * dx
x = np.linspace(np.min(end_points), np.max(end_points), 200)
plt.plot(x, np.exp(- x**2 / (2 * std**2)) / np.sqrt(2 * np.pi * std**2),
        label=r"Analytic Solution, Normal with $\sigma = \sqrt{N} \mathrm{d} x$")
plt.xlabel("end point")
plt.ylabel("count")
plt.title("Random Walk after %i steps using %i walks" % (steps, do_walks))
plt.legend()

plt.subplot(2,1,2)
plt.plot(dist)
plt.xlabel("step")
plt.ylabel("dist")
plt.title("Example Random Walk")
plt.tight_layout()
plt.show()
