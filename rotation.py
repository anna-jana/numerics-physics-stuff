import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

omega = 2*np.pi/20 # [rad/sec]
radius = 100.0 # [m]

T = 20.0 # [sec]
drops = 100
t = np.linspace(0, T, drops)

#v0 = np.repeat(np.reshape(np.array([1.0,0,0]), (1,3)), drops, axis=0)
v1 = radius*np.concatenate([np.zeros((drops, 1)),
                     np.reshape(np.cos(omega*t), (drops, 1)),
                     np.reshape(np.sin(omega*t), (drops, 1))],
                     axis=1)


x0 = radius*np.concatenate([np.zeros((drops, 1)),
               np.reshape(-np.sin(omega*t), (drops, 1)),
               np.reshape(np.cos(omega*t), (drops, 1))],
               axis=1)

v0 = -x0/radius*10

v = v0 + v1

t = np.repeat(np.reshape(t, (drops, 1)), 3, axis=1)
x = x0 + t*v

x, y, z = x[:,0], x[:,1], x[:,2]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
x, y, z = x0[:,0], x0[:,1], x0[:,2]
ax.plot(x, y, z)
plt.show()


