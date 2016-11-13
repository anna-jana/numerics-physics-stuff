import numpy as np
from scipy.constants import gravitational_constant as G
from scipy.integrate import odeint
import matplotlib.pyplot as plt

M = 5.975e24 # [kg]
R = 6.378e6 # [m]
v0 = np.sqrt(2*G*M/R) # [m/s]

T = 10000.0 # [s]
steps = 1000
t = np.linspace(0.0, T, steps)

def rhs(y, t):
    x, v = y[:2], y[2:]
    a = -G*M/np.linalg.norm(x)**3*x
    res = np.zeros(4); res[:2] = v; res[2:] = a
    return res

psi = np.linspace(0, 2*np.pi, 1000)
plt.plot(R*np.cos(psi), R*np.sin(psi), color="blue", label="Erde")

throws = 10

label_set = False

for theta in np.linspace(0, np.pi/2, throws):
    y0 = np.array([0.0, R, v0*np.cos(theta), v0*np.sin(theta)])
    ys = odeint(rhs, y0, t)
    x1, x2 = ys[:, 0], ys[:, 1]
    if label_set:
        plt.plot(x1, x2, color="red")
    else:
        plt.plot(x1, x2, color="red", label="Bahnkurve")
    y0 = np.array([0.0, R, v0*np.cos(theta), v0*np.sin(theta)])
    ys = odeint(rhs, y0, -t)
    x1, x2 = ys[:, 0], ys[:, 1]
    if label_set:
        plt.plot(x1, x2, color="green")
    else:
        plt.plot(x1, x2, color="green", label="Bahnkurve zurueck in der Zeit")
        label_set = True

plt.xlabel("x")
plt.ylabel("y")
plt.title("Bahnen von Koerpern mit Fluchtgeschwindigkeit")
plt.legend()
plt.show()

