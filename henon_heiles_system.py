import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def henon_heiles_rhs(t, s):
    x, y, px, py = s
    Fx = - x - 2*x*y
    Fy = - y - (x**2 - y**2)
    return px, py, Fx, Fy

def henon_heiles_system(initial_pos, initial_vel,
        time_span=100, num_samples=1000):
    sol = solve_ivp(henon_heiles_rhs, (0, time_span),
            tuple(initial_pos) + tuple(initial_vel),
            t_eval=np.linspace(0, time_span, num_samples), method="BDF")
    plt.plot(sol.y[0, :], sol.y[1, :])
    plt.plot([initial_pos[0]], [initial_pos[1]], "or")
    plt.title("Henon Heiles System")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return sol

if __name__ == "__main__":
    henon_heiles_system((0, 1), (0.01, 0))
