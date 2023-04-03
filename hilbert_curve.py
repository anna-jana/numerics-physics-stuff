# https://en.wikipedia.org/wiki/Hilbert_curve
import numpy as np
import matplotlib.pyplot as plt
import L_system

# rotate/flip a quadrant appropriately
def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

# convert (x,y) to d
def xy2d(n, x, y):
    d = 0
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s**2 * ((3 * rx) ^ ry)
        x, y = rot(n, x, y, rx, ry)
        s //= 2
    return d

# convert d to (x,y)
def d2xy(n, d):
    t = d
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

n = 2**5
ps = np.array([d2xy(n, d) for d in range(n**2)])
plt.figure(layout="constrained")
plt.subplot(1,2,1)
plt.plot(ps[:, 0], ps[:, 1])
plt.title("Hilbert Curve")
init = "A"
rules = {
    "A": "+BF-AFA-FB+",
    "B": "-AF+BFB+FA-",
}
steps = L_system.run_L_system(5, init, rules)
plt.subplot(1,2,2)
L_system.draw_L_system(steps, np.pi / 2)
plt.title("Hilbert Curve (L-System)")
plt.show()
