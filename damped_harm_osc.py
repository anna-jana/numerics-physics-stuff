import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def plot_damped_oscillator(m=1.0, k=1.0, b=1.0):
    """
    m = 1.0 # mass [kg]
    k = 1.0 # spring constant [N/m]
    b = 1.0 # friction constant [Ns/m = kgm/s^2*s/m = kg/s]
    """

    two_gamma = b/m
    omega_squared = k/m

    gamma = two_gamma/2
    omega = np.sqrt(omega_squared)

    if gamma == 0.0:
        osc_type = "non damped oscillator"
    elif gamma < omega:
        osc_type = "underdamped"
    elif gamma > omega:
        osc_type = "overdamped"
    else:
        osc_type = "Critically damped"

    def rhs(y, t):
        x, v = y[0], y[1]
        a = -two_gamma*v - omega_squared*x
        y_prime = np.array([v, a])
        return y_prime

    x0 = 10.0 # [m]
    v0 = 0.0 # [m/s]
    y0 = np.array([x0, v0])

    T = 20.0 # [s]
    steps = 1000
    ts = np.linspace(0, T, steps)

    ys = odeint(rhs, y0, ts)
    xs, vs = ys[:,0], ys[:,1]

    plt.subplot(2,1,1)
    plt.title(r"%s $\gamma = %f $, $\omega = %f $" % \
            (osc_type, gamma, omega))
    plt.plot(ts, xs, color="black")
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    plt.subplot(2,1,2)
    plt.plot(ts, vs, color="black")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.show()

#plot_damped_oscillator()
#plot_damped_oscillator(b=10.0)
plot_damped_oscillator(k=1/4.)
