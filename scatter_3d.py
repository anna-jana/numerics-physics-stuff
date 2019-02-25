from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 1000

x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)

ax.scatter(x, y, z, color="black", marker="*")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
