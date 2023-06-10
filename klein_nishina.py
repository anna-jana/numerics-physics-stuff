import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c

thetas = np.linspace(0, 2*np.pi, 500)
E_gammas = np.array([2.75, 601e3, 511e3, 1.46e6, 10e6]) # [eV]
epsilons = E_gammas * c.e.value / (c.m_e.value * c.c.value**2)
wavelength_rations = [
    1 / (1 + epsilon * (1 - np.cos(thetas)))
    for epsilon in epsilons
]
dsigma_dOmegas = [
    0.5 * ratio**2 * (ratio + 1 / ratio - np.sin(thetas)**2)
    for ratio in wavelength_rations
]


fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
for E_gamma, dsigma_dOmega in zip(E_gammas, dsigma_dOmegas):
    ax.plot(thetas, dsigma_dOmega,
            label=r"$E_\gamma = $" + f"{E_gamma:.2e} eV")
ax.legend(ncols=2, framealpha=1, loc="lower center")
plt.title("Klein-Nishina Crosssection: $\mathrm{d} \sigma / \mathrm{d} \Omega / r_e^2$")
plt.tight_layout()
plt.show()
