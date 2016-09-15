import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

plt.ion()

# lets define the van der pol oscillator!
# d^2x/tx^2 - mu(1 - x^2)*dx/dt + x = 0
# d^2x/tx^2 = mu(1 - x^2)*dx/dt - x
# a = dv/dt = d^2x/dt^2 = mu*(1 - x^2)*dx/dt - x = mu*(1 - x^2)v - x
# dx/dt = v
# dv/dt = mu*(1 - x^2)v - x

def get_vdp_accel(x, v):
    return mu*(1 - x**2)*v - x

def vdp_rhs(state, t, mu):
    x, v = state[0], state[1]
    a = get_vdp_accel(x, v)
    return np.array([v, a])

# for mu = 0 we get the harmonic oscillator for k = 1
# d^2x/tx^2 - 0(1 - x^2)*dx/dt + x = 0
# d^2x/tx^2 + x = 0
# -d^2x/tx^2 = x

# let's simulate the system
x0 = 2.0
v0 = 0.0
init_state = np.array([x0, v0])

T = 100.0
steps = 1000
ts = np.linspace(0, T, steps)

mu = 5.0

hist = odeint(vdp_rhs, init_state, ts, args=(mu,))
xs, vs = hist[:, 0], hist[:, 1]

# display the evolution of the system over time
plt.subplot(2,1,1)
plt.title("Van der Pol oscillator over time for mu={}".format(mu))
plt.plot(ts, xs)
plt.ylabel("x")
plt.subplot(2,1,2)
plt.xlabel("t")
plt.ylabel("dx/dt")
plt.plot(ts, vs)

# make a direction field [dx/dt, d^2x/dt^2]
res = 20
x = np.linspace(min(xs), max(xs), res)
v = np.linspace(min(vs), max(vs), res)
xx, vv = np.meshgrid(x, v)
aa = get_vdp_accel(xx, vv)
plt.title("Direction field of the van der pol oscillator with mu={}".format(mu))
plt.quiver(xx, vv, vv, aa)
plt.xlabel("x")
plt.ylabel("dx/dt")
plt.plot(xs, vs, label="from [{}, {}]".format(x0, v0))
plt.legend()
plt.grid()

# how does adding an epsilon value effect the system?
mu = 15.0
hist = odeint(vdp_rhs, init_state, ts, args=(mu,))
xs, vs = hist[:, 0], hist[:, 1]
eps = 1e-3
init_state_eps = np.array([x0 + eps, v0 + eps])
res_eps = odeint(vdp_rhs, init_state_eps, ts, args=(mu,))
xs_eps, vs_eps = res_eps[:, 0], res_eps[:, 1]
x_err = xs - xs_eps
v_err = vs - vs_eps
plt.subplot(2,1,1)
plt.title("Error evolution in the van der pol oscillator")
plt.plot(ts, x_err)
plt.ylabel("x - x'")
plt.subplot(2,1,2)
plt.plot(ts, v_err)
plt.ylabel("v - v'")
plt.xlabel("t")
# ok, we have some kind of oscillation going on
plt.plot(ts, xs)
plt.plot(ts, xs_eps) # almost equal




raw_input()
