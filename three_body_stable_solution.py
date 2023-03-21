# https://en.wikipedia.org/wiki/Three-body_problem#cite_ref-34
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

DIM = 2
N = 3
m = 1.0

def rhs(t, u):
    rs = u[:N*DIM].reshape(N, DIM)
    Fs = np.zeros((N, DIM))
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = rs[i] - rs[j]
                Fs[i] += - m / np.linalg.norm(diff)**3 * diff
    return np.hstack([u[N*DIM:], np.ravel(Fs)])

r1 = np.array((-0.97000436, 0.24308753))
r3 = -r1
r2 = np.array((0, 0))
v1 = v3 = np.array((0.4662036850, 0.4323657300))
v2 = np.array((-0.93240737, -0.86473146))
tspan = 10.0
sol = solve_ivp(rhs, (0, tspan), np.hstack([r1, r2, r3, v1, v2, v3]), dense_output=True)
# u[step, position/velocity, nth particle, coordinate]
u = sol.sol(np.linspace(0, tspan, 500)).T.reshape(-1, 2, N, DIM)

plt.figure()
for i in range(N):
    plt.plot(u[:, 0, i, 0], u[:, 0, i, 1], label=f"{i}th particle")
plt.xlabel("x")
plt.ylabel("y")
plt.title("stable periodic solution to the three body problem")
plt.legend()
plt.show()
