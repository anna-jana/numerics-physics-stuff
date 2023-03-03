import numpy as np, matplotlib.pyplot as plt

N = 100
L = 10
dx = L / N # periodic boundary conditions
u = np.random.uniform(0, 1, (N, N))
v = np.random.uniform(0, 1, (N, N))
x = np.arange(N) * dx
f = 0.04
k = 0.06
D_u = 0.01
D_v = 0.005
tspan = 10.0
dt = 0.001
nsteps = int(np.ceil(tspan / dt))

def laplace(f):
    dx2 = np.roll(f, -1, 0) - 2 * f + np.roll(f, +1, 0)
    dy2 = np.roll(f, -1, 1) - 2 * f + np.roll(f, +1, 1)
    return (dx2 + dy2) / dx**2

for i in range(nsteps):
    print(f"\r{i+1}/{nsteps}", flush=True, end="")
    A = u*v**2
    dudt = D_u * laplace(u) - A + f*(1 - u)
    dvdt = D_v * laplace(v) + A - (f + k)*v
    u += dt * dudt # forward euler
    v += dt * dvdt

plt.figure(figsize=(10,5))
plt.suptitle("Grey-Scott Equation")
plt.subplot(1,2,1)
plt.pcolormesh(x, x, u)
plt.xlabel("x")
plt.ylabel("y")
plt.title("u")
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(x, x, v)
plt.xlabel("x")
plt.ylabel("y")
plt.title("v")
plt.colorbar()
plt.show()
