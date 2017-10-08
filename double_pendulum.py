
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(y, t, m1, m2, l1, l2):
    g = 9.81
    theta1, theta2, dtheta1, dtheta2 = tuple(y)
    beta = np.cos(theta1 - theta2)
    M = m1 + m2
    alpha = -(l1*l2*m2*dtheta2**2*np.sin(theta1 - theta2) + g*M*l1*np.sin(theta1))
    alpha_tick = -(m2*theta1**2*l1 + l2*np.sin(theta2 - theta1) + g*m2*l2*np.sin(theta2))
    gamma = m1*M*l1**2*l2**2 - m1*m2*l1**2*l2**2*beta**2
    theta1_accel = (m1*l2**2*alpha - m1*l1*l2*beta*alpha_tick)/gamma
    theta2_accel = (M*l1**2*alpha_tick - m2*l1*l2*beta*alpha)/gamma
    return np.array([dtheta1, dtheta2, theta1_accel, theta2_accel])


y0 = np.array([np.pi/4, np.pi/8, 0, 0])
t = np.linspace(0, 20, 1000) # s
m1 = 2.0 # kg
m2 = 1.0 # kg
l1 = 1.0 # m
l2 = 0.5 # m
args = (m1, m2, l1, l2)
ys = odeint(rhs, y0, t, args=args)
theta1, theta2, theta1_dot, theta2_dot = ys[:,0], ys[:,1], ys[:,2], ys[:,3]

plt.plot(t, theta1, label=r"$\theta_1$")
plt.plot(t, theta2, label=r"$\theta_2$")
plt.title(r"Double pendulum with $m_1 = 2\mathrm{kg}$, $m_2 = 1\mathrm{kg}$, $l_1 = 1\mathrm{m}$, $l_2 = 0.5\mathrm{m}$")
plt.xlabel("Time [sec]")
plt.ylabel("Angle [Rad]")
plt.grid()
plt.legend()
plt.show()



