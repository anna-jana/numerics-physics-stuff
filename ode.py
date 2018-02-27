import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    h = t[1] - t[0]
    for i in range(t.size - 1):
        ys[i + 1, :] = ys[i,:] + h*f(t[i], ys[i,:])
    return ys

def heun(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    h = t[1] - t[0]
    for i in range(t.size - 1):
        k1 = f(t[i], ys[i,:])
        k2 = f(t[i] + h/2, ys[i,:] + h/2*k1)
        ys[i + 1, :] = ys[i,:] + h*(k1 + k2)/2
    return ys

def rk4(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0,:] = y0
    h = t[1] - t[0]
    for i in range(t.size - 1):
        k1 = f(t[i],       ys[i,:])
        k2 = f(t[i] + h/2, ys[i,:] + h/2*k1)
        k3 = f(t[i] + h/2, ys[i,:] + h/2*k2)
        k4 = f(t[i] + h,   ys[i,:] + h*k3)
        ys[i+1,:] = ys[i,:] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return ys

def sx(t,x0,v0,m,k):
    c = k/m
    B = np.sqrt(c)
    C = np.arctan(-v0/(x0*B))
    A = x0/np.cos(C)
    return A*np.cos(B*t + C)

T = 20
steps = 1000
t = np.linspace(0, T, steps)
k = 2.3
m = 1.2
c = k/m
x0 = 100.0
v0 = 1.2
y0 = np.array([x0, v0])
f = lambda t, y: np.array([y[1], -c*y[0]])

ys = forward_euler(f, y0, t)
xs = ys[:, 0]
vs = ys[:, 1]
plt.plot(t, xs, label="forward euler")

ys = heun(f, y0, t)
xs = ys[:, 0]
vs = ys[:, 1]
plt.plot(t, xs, label="heun")

ys = rk4(f, y0, t)
xs = ys[:, 0]
vs = ys[:, 1]
plt.plot(t, xs, label="runge kutta 4th order")

plt.plot(t, sx(t, x0, v0, m, k), "--", label="analytic")
plt.legend()
plt.title("ODE solvers for x'' = -k/m*x")
plt.xlabel("t")
plt.ylabel("x")
plt.show()
