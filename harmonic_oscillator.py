from __future__ import print_function, division
import scipy.integrate as solver
import matplotlib.pyplot as plt
import numpy as np

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

# what is the periode of our oscillator?
periode_numeric = 2*np.mean(np.diff(ts[:-1][np.sign(xs[:-1]) != np.sign(xs[1:])]))
periode_analytic = 2*np.pi/np.sqrt(k/m)
print("T_sym =", str(periode_analytic) + "s,", "T_num =", str(periode_numeric) + "s")

# mx'' = -k*x
# x'' = -k/m*x
# x'' = -cx
# c = k/m

# x(t) = A*cos(Bt + C)
# x'(t) = -AB*sin(Bt + C)
# x''(t) = -AB^2*cos(Bt + C)

# -B^2*A*cos(Bt + C) = -c*A*cos(Bt + C)
# B^2 = c
# B = sqrt(c)

# x(0) = x0
# x''(0) = -c*x0
# -AB^2*cos(B0 + C) = -c*x0
# A*c*cos(C) = c*x0
# A*cos(C) = x0
# A = x0/cos(C)

# x'(0) = v0
# v0 = -A*sqrt(c)*sin(C)
# A = -v0/(sqrt(c)*sin(C))

# x0/cos(C) = A = -v0/(sqrt(c)*sin(C))
# sin(C)/cos(C) = -v0/(x0*sqrt(c)) = tan(C)
# C = atan(-v0/(x0*sqrt(c)))

# A = x0/cos(C)
# A = x0/cos(atan(-v0/(x0*sqrt(c))))
# A = x0/sqrt(1/((-v0/(x0*sqrt(c)))^2 + 1))
# A = x0/sqrt(1/(v0^2/(x0^2*c) + 1))
#
#
# A = x0/sqrt(1/(v0^2/(x0^2*c) + 1))
# B = sqrt(c)
# C = atan(-v0/(x0*sqrt(c)))
# x(t) = x0/sqrt(1/(v0^2/(x0^2*c) + 1))*cos(sqrt(c)*t + atan(-v0/(x0*sqrt(c))))
#

def analytic_solution(t,x0,v0,m,k):
    c = k/m
    B = np.sqrt(c)
    C = np.arctan(-v0/(x0*B))
    A = x0/np.cos(C)
    return A*np.cos(B*t + C)

xs_analytic = analytic_solution(ts, x0, v0, m, k)

# plot stuff
plt.subplot(2,1,1)
plt.title("Position of the harmonic oscillator")
plt.plot(ts, xs, label="numeric")
plt.plot(ts, xs_analytic, "--", label="analytic")
plt.xlabel("t/s")
plt.ylabel("x/m")
plt.legend()

plt.subplot(2,1,2)
plt.plot(ts, vs)
plt.xlabel("t/s")
plt.ylabel("v/(m/s)")
plt.title("Velocity of the harmonic oscillator")
