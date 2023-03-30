from __future__ import print_function, division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.integrate as solver

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

fig = plt.figure(1)
ax = fig.add_subplot(projection="3d")
ax.plot(xs, ys, zs)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Lorenz Attractor")

# wie wirkt sich ein stoerung aus?
y0_prime = y0 + 1e-5
res_prime = solver.odeint(lorenz_rhs, y0_prime, ts)
xs_prime, ys_prime, zs_prime = res_prime[:,0], res_prime[:,1], res_prime[:,2]
x_err = xs - xs_prime
y_err = ys - ys_prime
z_err = zs - zs_prime

plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(ts, x_err)
plt.title("x")
plt.subplot(3, 1, 2)
plt.plot(ts, y_err)
plt.title("y")
plt.subplot(3, 1, 3)
plt.plot(ts, z_err)
plt.title("z")
plt.show()
