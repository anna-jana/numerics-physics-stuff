from __future__ import print_function, division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.integrate as solver

plt.ion()
plt.style.use("classic")

delta = 10.0
roh = 28.0
beta = 8/3

def lorenz_rhs(y, t):
    x, y, z = y[0], y[1], y[2]
    return np.array([
        delta*(y - x),
        x*(roh - z) - y,
        x*y - beta*z
    ])

time = 100.0
steps = 10000
ts = np.linspace(0, time, steps)
y0 = np.array([1.0, 1.0, 1.0])
res = solver.odeint(lorenz_rhs, y0, ts)
xs, ys, zs = res[:,0], res[:,1], res[:,2]

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(xs, ys, zs)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Lorenz Attractor")

raw_input()
