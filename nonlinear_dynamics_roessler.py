import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

def rhs(t, u, a, b, c):
    x, y, z = u
    return (-y - z, x + a*y, b + z*(x - c))

def compute_poincare_section(orbit, plane_base, plane_normal, tstep_prime, tspan):
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

def compute_max_lyapunov_exponent(rhs, params, x0, eps, dt, N, t_relax):
    sol = solve_ivp(rhs, (0, t_relax), x0, args=params)
    x0 = sol.y[:, -1]

    def advance(x):
        return solve_ivp(rhs, (0, dt), x, args=params)

    sol = advance(x0)
    tangent = sol.y[:, 0] - sol.y[:, 1]
    tangent /= np.linalg.norm(tangent)
    pertubation = np.random.randn(tangent.shape[0])
    pertubation /= np.linalg.norm(pertubation)
    pertubation = pertubation - np.dot(tangent, pertubation) * tangent
    pertubation *= eps

    s = 0.0

    for i in range(N):
        x_pertubation = x0 + pertubation
        pertubation_sol = advance(x_pertubation)
        sol = advance(x0)

        difference = pertubation_sol.y[:, -1] - sol.y[:, -1]

        local_lyapunov = np.log(np.linalg.norm(difference) / eps) / dt
        s += local_lyapunov

        pertubation = difference * eps / np.linalg.norm(difference)
        x0 = sol.y[:, -1]

    return s / N

def compute_delay_embedding(data, offsets, var_ids):
    max_offset = np.max(offsets)
    nsamples = data.shape[1]
    return [
        data[var_id, offset : nsamples - max_offset + offset]
        for var_id, offset in zip(var_ids, offsets)
    ]

if __name__ == "__main__":
    params = 0.2, 0.2, 5.7
    u0 = (1, 1, 1)
    tspan = 1000.0

    sol = solve_ivp(rhs, (0, tspan), u0, args=params, dense_output=True)
    data = sol.sol(np.linspace(0, tspan, 10000))

    plane_base = np.array((0,0,0))
    plane_normal = np.array((1, 0, 0))
    tstep_prime = 0.1
    ps = compute_poincare_section(sol.sol, plane_base, plane_normal, tstep_prime, tspan)

    first = ps[:-1, 1]
    second = ps[1:, 1]

    max_lyapunov_exponent = compute_max_lyapunov_exponent(rhs, params, u0, 1e-3, 0.1, 1000, 1000.0)

    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(*data, lw=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"maximal lyapumov exponent: {max_lyapunov_exponent:.2}")

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

    offsets = (0, 6, 17)
    a, b, c = compute_delay_embedding(data, offsets=offsets, var_ids=(0, 0, 0))
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(a, b, c, lw=0.1)
    ax.set_xlabel(f"x(t + {offsets[0]} * dt]")
    ax.set_ylabel(f"x(t + {offsets[1]} * dt]")
    ax.set_zlabel(f"x(t + {offsets[2]} * dt]")
    ax.set_title("delay embeeding of the x variable with dt = {")

    plt.show()
