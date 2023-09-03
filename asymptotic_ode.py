# asymptotic analysis of
# y'' = y / x^3
# as x -> 0
# idea (green): y = exp(S(x))
# y ~ exp(2 / sqrt(x) + 3/4 * ln(x) + K)
# with is WKB = for Q(x) = V - E = 1 / x^3
# y ~ K exp(+/- integral^x sqrt( Q(s) ) ds) / Q^(1/4))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

tspan = (1.0, 0.0)
# without r/atol arguments different integrators give different results and give artifacts at x = 0
sol = solve_ivp(lambda x, y: [y[1], y[0] / x**3], tspan, (1.0, 0),
        method="LSODA", t_eval=np.linspace(*tspan, 500), atol=1e-10, rtol=1e-10)
plt.plot(sol.t, sol.y[0, :] / (np.exp(2 / np.sqrt(sol.t)) * sol.t**(3/4)))
plt.xlabel("x")
plt.ylabel(r"$y_\mathrm{exact} / (\exp(2 / \sqrt{x}) x^{3/4})$")
plt.title(r"Asymptotic Analysis of $y'' = y / x^3$: $y \sim K \exp(2 / \sqrt{x}) x^{3/4}$")
plt.show()
