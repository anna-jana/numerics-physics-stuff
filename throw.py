import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import odeint

T = 20
omega = np.array([2*np.pi/T, 0, 0])
radius = 100.0
r0 = np.array([0, radius, 0])

def rhs(y, t):
    r, v = y[:3], y[3:]
    a_z = -np.cross(omega, np.cross(omega, r))
    a_c = -2*np.cross(omega, v)
    a = a_c + a_z
    ans = np.zeros(6); ans[:3] = v; ans[3:] = a
    return ans

fig = plt.figure()
ax = fig.gca(projection='3d')

label = False
for speed in np.linspace(1,100, 5):
    v0 = np.cross(omega, r0) - r0/radius*speed

    y0 = np.zeros(6); y0[:3] = r0; y0[3:] = v0

    sim_time = 2.0
    steps = 1000
    ts = np.linspace(0, sim_time, steps)

    ys = odeint(rhs, y0, ts)

    x, y, z = ys[:, 0], ys[:, 1], ys[:, 2]

    if label:
        ax.plot(x, y, z, color="red")
    else:
        ax.plot(x, y, z, color="red", label="wasser")
        label = True

phi = np.linspace(0, 2*np.pi, 1000)
ax.plot(np.zeros(1000), np.cos(phi)*radius, np.sin(phi)*radius, color="blue", label="Raumschiff")
plt.legend()

plt.show()
