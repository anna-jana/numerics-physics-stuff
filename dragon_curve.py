import numpy as np
import matplotlib.pyplot as plt

# multiplying my i or -i turns a complex number by 90 degrees
turn_left = 1j
turn_right = -1j
nsteps = 15
directions = np.array([turn_right])
for i in range(nsteps):
    # - turns R -> L and L -> R
    directions = np.hstack([directions, [turn_right], -directions[::-1]])

# first multily all direction changes to get an absolute direction in each step
# then sum them to get the positions along the curve
# the cum is to get very step
xs = directions.cumprod().cumsum()

plt.plot(xs.real, xs.imag)
plt.title("Dragon Curve")
plt.axis("off")
plt.show()
