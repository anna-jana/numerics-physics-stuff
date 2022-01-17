import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# d_t u + u d_x u = D d^2_x u
# d_t u_hat + fft(u d_x u) = - D k^2 u_hat

L = 2*np.pi
N = 256
D = 0.01
ts = [0.0, 0.5, 1.0, 1.5]
dt = 1e-4

dx = L/N
x = np.linspace(0, L, N)
u = np.sin(x)
u_hat = fft(u)
tspan = ts[-1]
k = np.array(list(range(0, N//2)) + [0] + list(range(-N//2 + 1, 0)))
us = []
index = 0
dudx = np.empty(len(u))
t = 0.0
steps = np.ceil(ts[-1] / dt)
i = 0

while t <= tspan:
    print("step =", i, "of", steps)
    dudx[1:-1] = (u[:-2] - u[2:]) / (2*dx)
    dudx[0] = 1/dx*(-3/2*u[0] + 2*u[1] - 0.5*u[2])
    dudx[-1] = -1/dx*(-3/2*u[-1] + 2*u[-2] - 0.5*u[-3])
    u_hat += dt * (- fft(dudx * u) - D * k**2 * u_hat)
    if t >= ts[index]:
        us.append(u)
        index += 1
    u = ifft(u_hat).real
    i += 1
    t += dt

for t, u in zip(ts, us):
    plt.plot(x, u, label="t = %.2f" % (t,))
plt.xlabel("x")
plt.ylabel("u")
plt.title("Burgers Equation $\partial_t u + u \partial_x u = D \partial^2_x u$ with D = %.2f" % (D,))
plt.legend()
plt.grid()
plt.show()
