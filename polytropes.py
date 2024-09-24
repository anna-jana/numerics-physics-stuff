import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# lame-emden equation
def rhs(z, y, n):
    w, dwdz = y
    return dwdz, - 2 * dwdz / z - w**n

def is_at_surface(t, y, _):
    w, _ = y
    return w
is_at_surface.terminal = True

fig, ax = plt.subplots(layout="constrained")

for n in range(0, 5 + 1):
    sol = solve_ivp(rhs, (1e-5, 5.0), [1.0, 0.0], args=(n,), events=[is_at_surface], dense_output=True)
    assert sol.success
    z = np.linspace(0, sol.t[-1], 400)
    rho_over_rho_center = sol.sol(z)[0, :]**n
    ax.plot(z, rho_over_rho_center, label=f"n = {n}")

ax.set_xlabel(r"$z = r / \alpha$, $\alpha = ((n + 1) / (4 \pi G) K \rho_c^{1/n - 1})^{1/2}$")
ax.set_ylabel(r"$\rho / \rho_c$")
ax.set_title(r"polytrope models $P = K \rho^\gamma$, $n = 1 / (\gamma - 1)$, central density $\rho_c$")
ax.legend(ncols=2)
plt.show()
# alpha =
