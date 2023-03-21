import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

def rhs(t, u, a, b, c):
    x, y, z = u
    return (-y - z, x + a*y, b + z*(x - c))

def poincare_section(orbit, plane_base, plane_normal, tstep_prime, tspan):
    nsteps = int(np.ceil(tspan / tstep_prime))
    tstep = tspan / (nsteps - 1)
    def plane_eval(p):
        return plane_normal @ (p - plane_base)
    crossings = []
    for i in range(nsteps):
        t1 = i * tstep
        t2 = t1 + tstep
        if np.sign(plane_eval(orbit(t1))) != np.sign(plane_eval(orbit(t2))):
            opt_ans = root_scalar(lambda t: plane_eval(orbit(t)),
                                bracket=(t1, t2))
            if opt_ans.converged:
                crossings.append(orbit(opt_ans.root))
    return np.array(crossings)

params = 0.2, 0.2, 5.7
u0 = (1, 1, 1)
tspan = 1000.0

sol = solve_ivp(rhs, (0, tspan), u0, args=params, dense_output=True)
orbit = sol.sol

plane_base = np.array((0,0,0))
plane_normal = np.array((1, 0, 0))
tstep_prime = 0.1
ps = poincare_section(sol.sol, plane_base, plane_normal, tstep_prime, tspan)

first = ps[:-1, 1]
second = ps[1:, 1]

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.plot(*sol.sol(np.linspace(0, tspan, 10000)), lw=0.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.figure(layout="constrained")
plt.scatter(ps[:, 1], ps[:, 2])
plt.xlabel("y")
plt.ylabel("z")
plt.title(f"poincare section with base = {plane_base} and"
            f"normal = {plane_normal}, i.e. the x plane")

plt.figure(layout="constrained")
plt.scatter(first, second)
plt.xlabel("first")
plt.ylabel("second")
plt.title("poincare map")

plt.show()
