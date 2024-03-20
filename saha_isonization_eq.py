import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

nL = 2.69e25 * u.m**(-3)
eps = 13.6 * u.eV
T = np.linspace(273, 200000, 500) * u.K
kBT = c.k_B * T
g_1 = 2
g_0 = 2
lamda = np.sqrt(c.h**2 / (2*np.pi * c.m_e * kBT))

for n_over_nL in (0.01, 1, 100):
    n = n_over_nL * nL
    A = 2 / (n*lamda**3) * g_1 / g_0 * np.exp(- eps / kBT)
    x = - A / 2 + np.sqrt((A / 2)**2 + A)
    plt.plot(T / 1e3, x, label=f"n = {n_over_nL} * n_L")

plt.legend()
plt.xlabel("temperature T / 1000 K")
plt.ylabel("ionization fraction x = n_1 / n")
plt.title("Saha Eq.")
plt.show()
