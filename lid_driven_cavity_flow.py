# solves the lid driven cavity flow problem in dimensionless variables
# of the incompressible navier stokes equations using a finite difference scheme
# in primitive variable form using a projection method for the pressure.

import matplotlib.pyplot as plt
import numpy as np

def solve_lid_driven_cavity_flow(N = 60, dt = 1e-3, tol = 1e-8, max_steps = 100, tspan = 1.0, Re = 100.0, U0 = 1.0, omega = 1.0):
    # Step -1: parameters
    # N + 1 are the number of grid points in each direction
    dx = 1 / N
    p0 = 1.0
    steps = int(np.ceil(tspan / dt))

    # Step 0. allocate memory and setup initial conditions v = 0, p = 1
    v = np.zeros((N + 1, N + 1, 2)) # x and y component of the velocity
    p = np.ones((N + 1, N + 1)) * p0
    tmp_v = np.zeros((N + 1, N + 1, 2))
    new_p = np.empty_like(p)
    new_p[0,:] = new_p[-1,:] = new_p[:,0] = new_p[:,-1] = p0
    force = None
    grad_p = np.empty((N - 1, N - 1, 2))

    # set initial conditions
    tmp_v[:, 0, 0] = v[:, 0, 0] = U0

    for step in range(steps):
        print("*** computing step {} out of {} steps ***".format(step, steps))
        # compute derivative of velocity without pressure gradient
        dvdx = (v[2:, 1:-1, :] - v[:-2, 1:-1, :]) / (2 * dx)
        dvdy = (v[1:-1, 2:, :] - v[1:-1, :-2, :]) / (2 * dx)
        new_NL = v[1:-1, 1:-1, 0][:, :, None] * dvdx + v[1:-1, 1:-1, 1][:, :, None] * dvdy
        d2vdx2 = (v[:-2, 1:-1, :] - 2 * v[1:-1, 1:-1, :] + v[2:, 1:-1, :]) / (dx**2)
        d2vdy2 = (v[1:-1, :-2, :] - 2 * v[1:-1, 1:-1, :] + v[1:-1, 2:, :]) / (dx**2)
        laplace = d2vdx2 + d2vdy2
        new_VIS = 1 / Re * laplace
        new_force = - new_NL + new_VIS

        # advance velocity without pressure gradient
        if step == 0:
            # euler
            tmp_v[1:-1, 1:-1, :] = v[1:-1, 1:-1, :] + dt * new_force
        else:
            # adams bashforth 2. order
            tmp_v[1:-1, 1:-1, :] = v[1:-1, 1:-1, :] + dt * 0.5 * (3 * new_force - force)
        # store new force as the new old one for the adams bashforth step
        force = new_force

        # compute the pressure with a poisson equation using an iterative method linear solver
        # (we use jacobi there bc. its easier to implement)
        # laplace p = div v
        d_tmp_vx_dx = (tmp_v[2:, 1:-1, 0] - tmp_v[:-2, 1:-1, 0]) / (2*dx)
        d_tmp_vy_dy = (tmp_v[1:-1, 2:, 1] - tmp_v[1:-1, :-2, 1]) / (2*dx)
        p_rhs = (d_tmp_vx_dx + d_tmp_vy_dy) / dt
        gauss_step = 1
        while True:
            new_p[1:-1, 1:-1] = (p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1]) / 4 - dx**2*p_rhs/4
            mse = np.sum((p - new_p)**2) / (N - 1)**2
            p = (1 - omega) * p + omega * new_p
            if mse < tol: break
            if not np.isfinite(mse): raise ValueError("jacobi iteration diverged")
            if gauss_step > max_steps: raise ValueError("jacobi iteration took too long")
            gauss_step += 1

        # compute the real velocity using the gradient of the pressure and set boundary conditions
        grad_p[:, :, 0] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dx)
        grad_p[:, :, 1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx)
        v[1:-1, 1:-1, :] = tmp_v[1:-1, 1:-1, :] + dt * (- grad_p)
    return v, p

N = 60
Re = 100.0
U0 = 1
x = y = np.linspace(0, 1, N+1)
v, p = solve_lid_driven_cavity_flow(N=N, Re=Re, U0=U0)
# Step 7. visualize the result
plt.contourf(x, y, np.linalg.norm(v, axis=2).T, 20)
plt.colorbar()
plt.streamplot(x, y, v[:, :, 0].T, v[:, :, 1].T)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lid Driven Cavity Flow, Re = 100, U0 = 1")
plt.show()
