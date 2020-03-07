import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(xs, t):
    xs = xs.reshape(xs.shape + (1,))
    t = t.reshape((1, 1, t.size))
    while xs.shape[0] > 1:
        between_xs = xs[1:,:,:] - xs[:-1,:,:]
        xs = xs[:-1,:,:] + t*between_xs
    return xs[0].transpose()

control_ps = np.array([[0.0, 0.0],
                       [0.4, 0.7],
                       [1.0, 0.0]])
n = 100
bezier_curve = bezier_curve(control_ps, np.linspace(0, 1, n))

plt.plot(bezier_curve[:,0], bezier_curve[:,1], "-b", label="Bezier Curve")
plt.plot(control_ps[:,0], control_ps[:,1], "ok", label="Control Points")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bezier Curve with three control points")
plt.grid()
plt.show()
