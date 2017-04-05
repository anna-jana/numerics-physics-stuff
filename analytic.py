import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def plot_anamech(V_fn, rhs, time=10,
        max_x=10, max_v=10, res=100,
        x0=None, v0=None,
        name=""):
    """
    DESCRIPTION:
        Plots the trajectory of a system in the phase space
        and there lagragian and hamiltonian.
    PARAMETER:
        V_fn: callable, calculates the potential energy of the system
        rhs: callable, calculates the right hand side of the equations of motion
        time: time to simulate the system (seconds)
        max_x: the absolute value of the minimum and maximum of the coordinate to display
        max_v: the absolute value of the minimum and maximum of the velocity to display
        res: number of grid points to plot the hamiltonian and lagragian
        x0: initial position
        v0: initial velocity
        name: name of the system to display
    """
    # compute lagragian and hamiltonian
    x = np.linspace(-max_x, max_x, res)
    v = np.linspace(-max_v, max_v, res)
    xx, vv = np.meshgrid(x, v)

    T = 0.5*m*vv**2
    V = V_fn(xx)
    L = T - V
    H = T + V

    # simulate motion
    t = np.linspace(0, time, 1000)

    if x0 is None:
        x0 = max_x/10*8
    if v0 is None:
        v0 = 0.0
    y0 = x0, v0
    ys = odeint(rhs, y0, t)
    xs, vs = ys[:,0], ys[:,1]

    # plot it
    plt.subplot(2,1,1)
    plt.ylabel("v[m/s]")
    plt.title("lagragian {}".format(name, m, k))
    plt.pcolormesh(xx, vv, L)
    plt.colorbar()
    plt.plot(xs, vs, color="black", label=r"$x_0 = {}, v_0 = {}$".format(x0, v0))
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel("x[m]")
    plt.ylabel("v[m/s]")
    plt.title("hamiltonian {}".format(name, m, k))
    plt.pcolormesh(xx, vv, H)
    plt.colorbar()
    plt.plot(xs, vs, color="black", label=r"$x_0 = {}, v_0 = {}$".format(x0, v0))
    plt.legend()
    plt.show()

######### examples #######
m = 1.0 # kg
M = 100 # kg
k = 1.0 # N/m
g = 9.81 # m/s^2

# harmonic oscillator
def osc_rhs(y, t):
    x, v = y
    a = -k/m*x
    return v, a

def osc_V(x):
    return 0.5*m*x**2

# free fall
def fall_rhs(y, t):
    x, v = y
    a = -g
    return v, a

def fall_V(x):
    return g*m*x

# gravity
def gravity_rhs(y, t):
    x, v = y
    a = -M*m/x**2
    return v, a

def gravity_V(x):
    return -M*m/abs(x)

if __name__ == "__main__":
    plot_anamech(osc_V, osc_rhs)
    plot_anamech(fall_V, fall_rhs, time=1)
    plot_anamech(gravity_V, gravity_rhs, time=2.4)
