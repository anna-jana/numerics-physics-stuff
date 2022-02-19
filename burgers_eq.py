import numpy as np, matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

# d_t u + u d_x u = D d^2_x u
# d_t u_hat + fft(u d_x u) = - D k^2 u_hat
# d_t u_hat + fft(1 / 2 * d_x u^2) = - D k^2 u_hat
# d_t u_hat - i k / 2 * fft((ifft(u))^2) = - D k^2 u_hat

L = 2*np.pi
N = 256
D = 0.01
ts = [0.0, 0.5, 1.0, 1.5]
dt = 1e-4

x = np.linspace(0, L, N)
u = np.sin(x)
u_hat = fft(u)
tspan = ts[-1]
k = fftfreq(x.size, x[1] - x[0]) * 2 * np.pi
us = []
index = 0
dudx = np.empty(len(u))
t = 0.0
steps = np.ceil(ts[-1] / dt)
i = 0

while t <= tspan:
    print("step =", i, "of", steps)
    u = ifft(u_hat).real
    u_hat += dt * (0.5j * k * fft(u**2) - D * k**2 * u_hat)
    if t >= ts[index]:
        us.append(u)
        index += 1
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
