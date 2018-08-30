
# coding: utf-8

# # Michaels Menten Kinetik

# $$
# \frac{d[P]}{dt} = k_{cat}[ES]
# $$
# $$
# \frac{d[S]}{dt} = k_r[ES] - k_v[E][S]
# $$
# $$
# \frac{d[E]}{dt} = -k_v[E][S] + k_r[ES] + k_{cat}[ES]
# $$
# $$
# \frac{d[ES]}{dt} = k_v[E][S] - k_r[ES] - k_{cat}[ES]
# $$

import numpy as np
import scipy.integrate as solver

import matplotlib.pyplot as plt
import matplotlib

# k_r schnell: komplex zerfaellt langsamer als es sich bildet
# k_v schnell: komplex bildet sich schneller als es zerfaellt
# k_cat:       produkte reaktion ist langsam
# P0:          keine produkte
# S0:          viel edukte
# E0:          ein bisschen enzym
# ES0:         noch keine komplexe

def rhs(y, t, k_r, k_v, k_cat):
    P = y[0]
    S = y[1]
    E = y[2]
    ES = y[3]
    return np.array([
        k_cat*ES,
        k_r*ES - k_v*E*S,
        -k_v*E*S + k_r*ES + k_cat*ES,
        k_v*E*S - k_r*ES - k_cat*ES])

P0 = 50.0
S0 = 50.0
E0 = 50.0
ES0 = 50.0
k_r = 5.0
k_v = 5.0
k_cat = 5.0
y0 = np.array([P0, S0, E0, ES0])

T = 2.0
steps = 1000
ts = np.linspace(0, T, steps)

ys = solver.odeint(rhs, y0, ts, args=(k_r, k_v, k_cat))
Ps = ys[:, 0]
Ss = ys[:, 1]
Es = ys[:, 2]
ESs = ys[:, 3]

plt.plot(ts, Ps, label="[P]")
plt.plot(ts, Ss, label="[S]")
plt.plot(ts, Es, label="[E]")
plt.plot(ts, ESs, label="[ES]")
plt.legend()
plt.xlabel("t/s")
plt.ylabel("c/mol")
plt.show()

