from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

Electrode = namedtuple("Electrode", ["x", "y", "r", "V"])

@jit
def setup(electrodes, dx, N):
    is_boundary = np.empty((N, N), dtype="bool")
    V = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            boundary_value = None
            if i == 0 or j == 0 or i == N - 1 or j == N - j:
                is_boundary[i][j] = True
                boundary_value = 0.0
            else:
                is_boundary[i][j] = False
                for e in electrodes:
                    x = i * dx
                    y = j * dx
                    if (x - e.x)**2 + (y - e.y)**2 < e.r**2:
                        is_boundary[i][j] = True
                        boundary_value = e.V
                        break
            V[i][j] = boundary_value if is_boundary[i][j] else 0
    return V, is_boundary

@jit
def SOR(V, is_boundary, eps, alpha, dx):
    N = V.shape[0]
    steps = 0
    while True:
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if not is_boundary[i][j]:
                    V_star = (V[i + 1][j] + V[i - 1][j] + V[i][j - 1] + V[i][j + 1]) / 4.0
                    Delta_V = V_star - V[i][j]
                    V[i][j] += alpha * Delta_V
        bad = False
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if not is_boundary[i][j]:
                    laplace = (V[i - 1][j] - 2 * V[i][j] + V[i + 1][j]) / dx**2 + (V[i][j - 1] - 2 * V[i][j] + V[i][j + 1]) / dx**2
                    if abs(laplace) >= eps:
                        bad = True
                        break
            if bad: break
        steps += 1
        if not bad: return steps


N = 100
L = 1.0
dx = L / (N - 1)
eps = 1e-5
alpha = 1.7

electrodes = [
    Electrode(0.2, 0.2, 0.1, +1),
    Electrode(0.4, 0.2, 0.1, -1),
    Electrode(0.7, 0.3, 0.1, +1),
    Electrode(0.5, 0.8, 0.1, +1),
]

V, is_boundary = setup(electrodes, dx, N)
steps = SOR(V, is_boundary, eps, alpha, dx)
print(steps, "steps needed")

x = y = np.linspace(0, L, N)
plt.pcolormesh(x, y, V)
plt.xlabel("x")
plt.ylabel("y")
plt.title("V")
plt.colorbar()
plt.show()
