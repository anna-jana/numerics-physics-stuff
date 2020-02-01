import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation_ftcs(D, L, T_initial, T_left, T_right, dt, tspan):
    N = T_initial.size + 1
    t = np.arange(0, tspan, dt)
    x = np.linspace(0, L, N + 1)
    T = np.empty((len(t), N + 1))
    dx = L / N
    s = dt * D / dx**2
    T[0, 1:-1] = T_initial
    T[0] = T_left
    T[-1] = T_right
    for i in range(len(t) - 1):
        T[i + 1, 0] = T_left
        T[i + 1, -1] = T_right
        T[i + 1, 1:-1] = T[i, 1:-1] + s * (T[i, :-2] - 2 * T[i, 1:-1] + T[i, 2:])
    return t, x, T

if __name__ == "__main__":
    N = 120
    D = 1.0
    t, x, T = solve_heat_equation_ftcs(D, 1.0, np.zeros(N - 1), 0.0, 1.0, 1e-5, 0.1)
    plt.pcolormesh(x, t, T)
    plt.colorbar()
    plt.xlabel("x [m]")
    plt.ylabel("t [s]")
    plt.title("D = " + str(D))
    plt.show()
