"""
Numerical integration of the SIR disease model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# right hand side of the equation
def compute_SIR_rhs(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = - beta * I * S / N # number of uninfected people
    dIdt = beta * I * S / N - gamma * I # number of infected people
    dRdt = gamma * I # number of recovert people
    return dSdt, dIdt, dRdt

# parameters
beta = 1 / 1.0 # = 1 / time between contacts between people
N = 100 # = number of people
gamma = 1 / 14.0 # = 1 / time until recovery
args = (N, beta, gamma)
initial_infected_ratio = 1e-4
num_infected = initial_infected_ratio * N
y0 = (N - num_infected, num_infected, 0)

if __name__ == "__main__":
    # integrate the system
    t_s = np.linspace(0, 8 / gamma, 400)
    S, I, R = odeint(compute_SIR_rhs, y0, t_s, args).T

    # plot the result
    plt.plot(t_s, S, label="S")
    plt.plot(t_s, I, label="I")
    plt.plot(t_s, R, label="R")
    plt.xlabel("t")
    plt.ylabel("number of people")
    plt.title(r"SIR disease model with $N = %i, \beta = %.2f, \gamma = %.2f$" % args)
    plt.text(t_s[len(t_s) // 2], N / 2,
            r"$\frac{\mathrm{d} S}{\mathrm{d} t} = - \beta I S / N$" + "\n" +
            r"$\frac{\mathrm{d}I}{\mathrm{d}t} = \beta I S / N - \gamma I$" + "\n" +
            r"$\frac{\mathrm{d}R}{\mathrm{d}t} = \gamma I$")
    plt.legend()
    plt.show()
