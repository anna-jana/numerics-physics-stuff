import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def hubble(t): return 1 / (2*t)
def inv_hubble(H): return 1 / (2*H)
def pot_deriv(theta, m): return m**2 * theta

def rhs(t, y, *args):
    theta, theta_dot = y
    theta_dot2 = - 3 * hubble(t) * theta_dot - pot_deriv(theta, *args)
    return theta_dot, theta_dot2

if __name__ == "__main__":
    m = 1e-1
    theta0 = 1.0
    tspan = (1e-3, 1000.0)
    sol = solve_ivp(rhs, tspan, [theta0, 0], t_eval=np.geomspace(*tspan, 1000), args=(m,))
    plt.plot(sol.t, sol.y[0])
    plt.axvline(inv_hubble(m), ls="--", color="black")
    plt.xscale("log")
    plt.xlabel("$t$")
    plt.ylabel(r"$\theta$")
    plt.show()
