import matplotlib.pyplot as plt
import numpy as np

N_f = 6
beta0 = 11 - 2 / 3 * N_f
Lambda = 200
k = np.linspace(2 * Lambda, 1e2*Lambda, 500)
alpha_s = 1 / (beta0 * np.log(k**2 / Lambda**2))

m = 1e-10
alpha_e_0 = 1 / 137
alpha_e = alpha_e_0 / (1 - alpha_e_0 / (3*np.pi) * np.log(k**2 / m**2))

plt.figure(constrained_layout=True)
plt.subplot(2,1,1)
plt.loglog(k**2, alpha_s, label="QCD")
plt.legend()
plt.xlabel(r"$k^2 / \mathrm{MeV}^2$")
plt.ylabel(r"$\alpha_s(k^2)$")
plt.subplot(2,1,2)
plt.loglog(k**2, alpha_e, label="QED")
plt.xlabel(r"$k^2 / \mathrm{MeV}^2$")
plt.ylabel(r"$\alpha_e(k^2)$")
plt.legend()
plt.suptitle("Running of Couplings")
plt.show()
