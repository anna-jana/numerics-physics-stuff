import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection="3d")

x = y = np.linspace(-2, 2, 100)
xx, yy = np.meshgrid(x, y)
rr = np.sqrt(xx**2 + yy**2)
zz = rr**4 - 7*rr**2
surf = ax.plot_surface(xx, yy, zz, cmap="coolwarm")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

#fig.colorbar(surf, orientation="horizontal")
fig.colorbar(surf)

plt.show()

