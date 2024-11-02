# https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/pdf/jphysiol01442-0106.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# there is more negative charge inside the cell membrane than outside
# changes are from K+ and Na+ ions moving through ion channels

# V: membrane potential (electric potential across the membrane)
# n: propability for a single gate in a K+ channel to be open
# C: capacitance of the membrane
# I_K: current of K+ ions
# I_Na: current of Na+ ions
# I_l: leakage current
# g_K: maximal conductance of the K+ channel
# g_Na: maximal conductance of the Na+ channel
# g_l: conductace of the leakage current
# E_K: equilibrium potential (equilibrium between diffusion and coloumb attraction) of K+ iosn
# E_Na: equilibrium potential (equilibrium between diffusion and coloumb attraction) of Na+ iosn
# E_l: equilibrium potential (equilibrium between diffusion and coloumb attraction) of leakage current
# alpha_n: rate of opening a gate in the K+ channel
# beta_n: rate of closing a gate in the K+ channel
# alpha_m: rate of opening a gate I in the Na+ channel
# alpha_j: rate of opening a gate II in the Na+ channel
# beta_m: rate of closing a gate I in the Na+ channel
# beta_j: rate of closing a gate II in the Na+ channel

def alpha_n(V):
    return 0.01 * (V + 10.0) / (np.exp((V + 10.0) / 10.0) - 1.0)

def beta_n(V):
    return 0.125 * np.exp(V / 80.0)

def alpha_m(V):
    return 0.1 * (25.0 - V) / (np.exp((25.0 - V) / 10.0) - 1.0)

def beta_m(V):
    return 4.0 * np.exp(- V / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(- V / 20.0)

def beta_h(V):
    return 1 / (np.exp((30.0 - V) / 10.0) + 1.0)

tmax = 20.0

C = 1.0
g_K =  36.0
g_Na = 120.0
g_l = 0.3
E_K = 115
E_Na = 50
E_l = -10.613

V0 = 50.0
n0 = 1.0
m0 = 0.0
h0 = 0.

def rhs(t, y, C, g_K, g_Na, g_l, E_K, E_Na, E_l):
    V, n, m, h = y
    I_K = g_K * n**4 * (E_K - V)
    I_Na = g_Na * m**3 * h * (E_Na - V)
    I_l = g_l * (E_l - V)
    dVdt = (I_K + I_Na + I_l) / C
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    return (dVdt, dndt, dmdt, dhdt)

sol = solve_ivp(rhs, (0, tmax), [V0, n0, m0, h0],
        args=(C, g_K, g_Na, g_l, E_K, E_Na, E_l))
assert sol.success

fig, axs = plt.subplots(2, 1, layout="constrained")
axs[0].plot(sol.t, sol.y[0])
axs[0].set_xlabel("t")
axs[0].set_ylabel("V")
axs[1].plot(sol.t, sol.y[1], label="n")
axs[1].plot(sol.t, sol.y[2], label="m")
axs[1].plot(sol.t, sol.y[3], label="h")
axs[1].set_xlabel("t")
axs[1].legend(loc=1, framealpha=1.0)
fig.suptitle("Hogkin Huxley Model")
plt.show()
