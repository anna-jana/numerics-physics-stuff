import matplotlib.pyplot as plt
import numpy as np

def velocity_verlet(F_fn, m, x0, v0, T, dt, args=(), kwargs={}):
    t = np.arange(0, T, dt)
    x = np.empty((len(t), len(x0)))
    v = np.empty_like(x)
    x[0, :] = x0
    v[0, :] = v0
    F = F_fn(x[0, :], *args, **kwargs)
    for i in range(len(t) - 1):
        x[i + 1, :] = x[i, :] + dt*v[i, :] + 0.5*dt**2*F/m
        new_F = F_fn(x[i + 1, :], *args, **kwargs)
        v[i + 1, :] = v[i, :] + (F + new_F)/2.0*dt
        F = new_F
    return t, x, v

def FPUT_rhs(x, alpha):
    left = x[:-2]
    right = x[2:]
    center = x[1:-1]
    F = np.empty_like(x)
    F[1:-1] = (left - 2*center + right)*(1 + alpha*(left - right))
    F[0] = F[-1] = 0.0 # the ends of the chain are fixed
    return F

if __name__ == "__main__":
    m = 1.0
    alpha = 0.1
    x0 = np.random.rand(10)
    x0[0] = x0[-1] = 0.0
    v0 = np.zeros(len(x0))
    t, x, v = velocity_verlet(FPUT_rhs, m, x0, v0, 100.0, 0.01, args=(alpha,))

    for i in range(len(x0)):
        plt.plot(t, x[:, i], label=f"i = {i}")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.title("Simulation of the FPUT system\n"
              + r"$m \ddot{x_j} = (x_{j - 1} - 2 x_j + x_{j + 1})(1 + \alpha (x_{j + 1} - x_{j - 1}))$"
              + f"\nwith $\\alpha = {alpha}$ and $m = {m}$")
    plt.tight_layout()
    plt.show()
