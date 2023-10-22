import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


def rhs(_x, y):
    phi, phi_prime = y
    return phi_prime, phi * (phi**2 - 1)


def bc(ya, yb):
    phi_a, phi_dot_a = ya
    phi_b, phi_dot_b = yb
    return (
        (phi_a + 1.0)**2,
        (phi_b - 1.0)**2,
    )


r_max = 2e1
xs = np.linspace(0.0, r_max, 400)
center = 0.9 * r_max
width = 0.1
ys_guess = np.vstack([
    np.tanh((xs - center) / (width * r_max)),
    1 / (width * r_max * np.cosh((xs - center) / (width * r_max))**2),
])
S = np.array([[0.0, 0.0], [0.0, 2.0]])

sol = solve_bvp(rhs, bc, xs, ys_guess, S=S, max_nodes=10**4)
print(sol.success)

plt.plot(sol.x, sol.y[0])
# for negative r to make it symmetrical
plt.plot(-sol.x, sol.y[0], color="tab:blue")
plt.xlabel(r"radius $r \lambda f^2$")
plt.ylabel(r"field $\phi / f$")
plt.title(
    r"spherical bubble with potential $V = \lambda / 4 (\phi^2 - f^2)^2$" +
    f" of radius $r \\lambda f^2$ = {r_max}")
plt.show()
