
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.constants import g

#
# Fandenpendel:
# $$ \ddot{\theta} = -k^2 \sin(\theta) $$
#
# Als Näherung:
# Harmonischer Oszillator:
# $$ \ddot{\theta} = -k^2 \theta  $$
#
# mit
# $$
# \theta(0) = \theta_0, \, \dot{\theta} = 0, \, k^2 = \frac{g}{l}
# $$

# Allgemeine Lösung des harmonischen Oszillators:
# $$
# \theta(t) = A e^{ikt} + B e^{-ikt}
# $$
# Anfandsbedinungen einsetzen:
# $$
# \theta(t) = \theta_0 \cos(kt)
# $$
# und
# $$
# \dot{\theta}(t) = - \theta_0 k \sin(kt)
# $$

steps = 1000 # [1]
T = 8.0 # [s]
t = np.linspace(0, T, steps) # [s]
v0 = 0.0 # [m/s]
l = 1.0 # [m]
k_sq = g / l
k = np.sqrt(k_sq)


def rhs(y, t):
    theta, v = y[0], y[1]
    a = - k_sq * np.sin(theta)
    return np.array([v, a])


def compare_harm_osc_and_pendulum(theta0_deg):
    theta0 = np.deg2rad(theta0_deg)

    theta_harm_osc = theta0 * np.cos(k * t)
    v_harm_osc = - theta0 * k * np.sin(k * t)

    y0 = np.array([theta0, v0])
    ys_pendel = odeint(rhs, y0, t)
    theta_pendel = ys_pendel[:, 0]
    v_pendel  = ys_pendel[:, 1]

    plt.subplot(2, 1, 1)
    plt.title(r"Harmonic Oscillator vs. Pendulum ($\theta_0 = %.2f^\circ$)" % theta0_deg)
    plt.plot(t, theta_harm_osc, "k", label="Harmonic Oscillator")
    plt.plot(t, theta_pendel, "--r", label="Pendulum")
    plt.plot(t, (theta_harm_osc - theta_pendel), "b", label="Signed Error")
    plt.grid()
    plt.xlabel("time t [s]")
    plt.ylabel(r"angle $\theta$ [rad]")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(theta_harm_osc, v_harm_osc, "k", label="Harmonic Oscillator")
    plt.plot(theta_pendel, v_pendel, "r", label="Pendulum")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.grid()
    plt.legend()

    plt.tight_layout()


plt.figure(1, figsize=(8,8))
compare_harm_osc_and_pendulum(5)

plt.figure(2, figsize=(8,8))
compare_harm_osc_and_pendulum(60)

plt.figure(3, figsize=(8,8))
compare_harm_osc_and_pendulum(170)
plt.show()
