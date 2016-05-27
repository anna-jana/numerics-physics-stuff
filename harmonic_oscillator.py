from __future__ import print_function, division
import scipy.integrate as solver
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
plt.style.use("ggplot")

# initial conditions and parameters
k = 1.0 # N = kg*m/s^2 = k*m => k = kg/s^2
m = 1.0 # kg
time = 100.0 # s
steps = 1000
x0 = 10.0 # m
v0 = 0.0 # m/s
ts = np.linspace(0, time, steps) # s
y0 = np.array([x0, v0]) # [m, m/s]

# the ode rhs
def rhs(y, t):
    """
    y = [x, v]
    y' = [v, -k*x]
    """
    return np.array([y[1], -k/m*y[0]])

# let's solve the harmonic oscillator numerically!
ys = solver.odeint(rhs, y0, ts)
xs = ys[:, 0]
vs = ys[:, 1]
plt.subplot(2,2,1)
plt.title("Position of the harmonic oscillator")
plt.plot(ts, xs)
plt.xlabel("t/s")
plt.ylabel("x/m")
plt.subplot(2,1,2)
plt.plot(ts, vs)
plt.xlabel("t/s")
plt.ylabel("v/(m/s)")
plt.title("Velocity of the harmonic oscillator")

# what is the periode of our oscillator?
periode_numeric = 2*np.mean(np.diff(ts[:-1][np.sign(xs[:-1]) != np.sign(xs[1:])]))
periode_analytic = 2*np.pi/np.sqrt(k/m)
print("T_sym =", str(periode_analytic) + "s,", "T_num =", str(periode_numeric) + "s")

# the analytic solution of the harmonic oscillator ODE
amp = np.sqrt(m/k*v0**2 + x0**2)
angle_vel = np.sqrt(k/m)
phase = np.arccos(x0/amp)
xs_analytic = amp*np.cos(angle_vel*ts + phase)
plt.plot(ts, xs, label="numeric")
plt.plot(ts, xs_analytic, label="analytic")
plt.xlabel("t/s")
plt.ylabel("x/m")
plt.title("numerical vs. analytical solution")
plt.legend()

raw_input()
