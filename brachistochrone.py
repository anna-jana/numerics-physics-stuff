import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def goal(y, l, h):
    theta_end, k = y
    return (
        0.5 * k**2 * (theta_end - np.sin(theta_end)) - l,
        0.5 * k**2 * (1 - np.cos(theta_end)) - h,
    )
l = 2.0
h = 1.0
sol = root(goal, (2*np.pi, 1.0), args=(l, h))
theta_end, k = sol.x

theta = np.linspace(0, theta_end, 400)
x = 0.5 * k**2 * (theta - np.sin(theta))
y = - 0.5 * k**2 * (1 - np.cos(theta))

plt.figure()
plt.plot(x, y)
plt.plot([0, l], [0, -h], "or")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Brachistochrone")
ax = plt.gca()
plt.show()
