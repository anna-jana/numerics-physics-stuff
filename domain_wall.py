import numpy as np
import matplotlib.pyplot as plt

# solution to the klein gordon eq. with potential = lambda / 4 (phi^2 - sigma^2)^2
# for d phi / dt = 0 and phi -> +/-sigma for x -> +/-inf:

# d^2 phi / dt^2 - d^2 phi / dx^2 = - lambda phi (phi^2 - sigma^2)
# d phi / dt = 0
# d^2 phi dx^2 = - lambda phi (phi^2 - sigma^2)
# 0.5 (d phi / dx)^2 = - lambda (phi^2 - sigma^2)^2 / 4
# d phi / dx = +/- sqrt(lambda / 2) (phi^2 - sigma^2)
# 1 / (phi^2 - sigma^2) d phi = +/- sqrt(lambda / 2) dx
# int 1 / (phi^2 - sigma^2) d phi = +/- sqrt(lambda / 2) dx
# |phi| < |sigma|
# - 1 / sigma * atanh(phi / sigma) + C = +/- sqrt(lambda / 2) * x
# atanh(phi / sigma) = sigma * (+/- sqrt(lambda / 2) x + C)
# phi = sigma * tanh(sigma * (+/- sqrt(lambda / 2) x + C))
# phi -> +/-sigma for x -> +/-inf ===> C = 0, + sign
# phi = sigma * tanh(x / (1/sqrt(lambda / 2) / sigma))
# Delta = 1/sqrt(lambda / 2) / sigma
# phi = sigma * tanh(x / Delta)

# energy density:
# T^{mu, nu} = d_mu phi d_nu phi - L g_{mu nu}
# T^{0 0} = 0.5 (dphi/dt)^2 - ((d phi / dt)^2 - (d phi / d x)^2 - lambda / 4 (phi^2 - sigma^2)^2)
# d phi / dt = 0 ==> T^{0, 0} = 0.5 (d phi / d x)^2 + lambda / 4 (phi^2 - sigma^2)^2
# d phi / d x = sigma / Delta cosh^-2(x / Delta)
# T^{0, 0} = 0.5 (sigma / Delta cosh^-2(x / Delta))^2 + lambda / 4 (sigma^2 tanh^2(x / Delta) - sigma^2)^2
#            0.5 (sigma^2 sqrt(lambda / 2) cosh^-2(x / Delta))^2 + lambda / 4 * sigma^4 (tanh^2(x / Delta) - 1)^2
#            sigma^4 lambda / 4 cosh^-4(x / Delta) + lambda / 4 * sigma^4 (cosh^-2(x / Delta))^2
#            sigma^4 lambda / 4 cosh^-4(x / Delta) + lambda / 4 * sigma^4 cosh^-4(x / Delta)
#            sigma^4 lambda / 2 cosh^-4(x / Delta)

x_range = 10.0
x = np.linspace(-x_range, x_range, 500)
sigma = 1.0

plt.figure()
ax = plt.gca()
ax2 = ax.twinx()

for lam in [0.1, 1, 5]:
    Delta = (lam / 2)**(-1/2) / sigma # width of the domain wall
    phi = sigma * np.tanh(x / Delta) # field value
    rho = lam / 2 * sigma**4 * np.cosh(x / Delta)**-4 # energy density

    p, = ax.plot(x, phi, label= r"$\lambda = $" + f"{lam:.2f}, " + r"$\Delta =$" + f"{Delta:.2f}")
    ax2.plot(x, rho, color=p.get_color(), ls="--")

ax.legend()
plt.xlabel("x")
ax.set_ylabel(r"field $\phi(x)$ / a.u., solid lines")
ax2.set_ylabel(r"energy density $\rho(x)$ / a.u., dashed lines")
plt.title(r"Domain Wall: $- \partial_z^2 \phi + \lambda \phi (\phi^2 - \sigma^2)$, " + r"$\sigma =$" + f"{sigma:.2f}")
plt.show()
