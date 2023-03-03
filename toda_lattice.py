import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rhs(t, y):
    N = y.size // 2
    q, p = y[:N], y[N:]
    dpdt = np.exp(- (q - np.roll(q, 1))) - np.exp(- (np.roll(q, -1) - q))
    dqdt = p
    return np.hstack([dqdt, dpdt])

tspan = 80.0
N = 100
np.random.seed(42)
y0 = np.hstack([np.random.uniform(-1, 1, N), np.zeros(N)])
sol = solve_ivp(rhs, (0, tspan), y0, dense_output=True)
ts = np.linspace(0, tspan, 100)
qs = sol.sol(ts)[:N, :].T
plt.figure()
plt.pcolormesh(np.arange(N), ts, qs)
plt.xlabel("lattice side $i$")
plt.ylabel("time $t$")
plt.colorbar(label="excitation of side $i$, $q_i$")
plt.title("Todo Lattice")
plt.show()
