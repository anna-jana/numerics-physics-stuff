# solves the lid driven cavity flow problem in dimensionless variables
# of the incompressible navier stokes equations using a finite difference scheme
# in primitive variable form using a projection method for the pressure.

import matplotlib.pyplot as plt
import numpy as np

def solve_lid_driven_cavity_flow(N = 60, dt = 1e-3, tol = 1e-10, max_steps = 100, tspan = 3.0, Re = 100.0, U0 = 1.0, omega = 1.0):
    # Step -1: parameters
    # N + 1 are the number of grid points in each direction
    dx = dy = 1 / N
    p0 = 1.0
    steps = int(np.ceil(tspan / dt))

    # Step 0. allocate memory and setup initial conditions v = 0, p = 1
    u = np.zeros((N + 1, N + 1)) # x component of the velocity
    v = np.zeros((N + 1, N + 1)) # y component of the velocity
    p = np.ones((N + 1, N + 1))
    tmp_v = np.zeros((N + 1, N + 1))
    tmp_u = np.zeros((N + 1, N + 1))
    p_rhs = np.empty_like(p)
    d_tmp_u_dx = np.empty_like(u)
    d_tmp_v_dy = np.empty_like(v)
    new_p = np.empty_like(p)
    force_x = force_y = None

    # set initial conditions
    tmp_u[0,:] = u[0,:] = U0

    for step in range(steps):
        print("*** computing step {} out of {} steps ***".format(step, steps))
        # Step 1. compute new derivatives of the velocity
        # Step 1.1 first derivatives of u, v in x and y direction
        dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        # Step 1.2 second derivatives of u, v in x and y direction
        d2udx2 = (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / (dx**2)
        d2udy2 = (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / (dy**2)
        d2vdx2 = (v[1:-1, :-2] - 2 * v[1:-1, 1:-1] + v[1:-1, 2:]) / (dx**2)
        d2vdy2 = (v[:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / (dy**2)
        # laplace of u and v
        laplace_u = d2udx2 + d2udy2
        laplace_v = d2vdx2 + d2vdy2

        # Step 2. compute new VIS and NL terms and the resulting force
        # (v dot nabla) v = (v_x * d_x + v_y * d_y) v = [v_x * d_x v_x + v_y * d_y v_x, v_x * d_x v_y + v_y * d_y v_y]
        new_NL_x = u[1:-1,1:-1] * dudx + v[1:-1,1:-1] * dudy
        new_NL_y = u[1:-1,1:-1] * dvdx + v[1:-1,1:-1] * dvdy
        new_VIS_x = 1 / Re * laplace_u
        new_VIS_y = 1 / Re * laplace_v
        new_force_x = - new_NL_x + new_VIS_x
        new_force_y = - new_NL_y + new_VIS_y

        # Step 3. compute temporal velocity field using the Adams Bashforth method of second order
        #         / euler for the first step
        if step == 0:
            # euler
            tmp_u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * new_force_x
            tmp_v[1:-1,1:-1] = v[1:-1,1:-1] + dt * new_force_y
        else:
            # adams bashforth 2. order
            tmp_u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * 0.5 * (3 * new_force_x - force_x)
            tmp_v[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * 0.5 * (3 * new_force_y - force_y)

        # store new force as the new old one for the adams bashforth step
        force_x = new_force_x
        force_y = new_force_y

        # Step 4. compute the pressure with a poisson equation using an iterative method linear solver
        #         and finite differences
        # Step 4.1 compute divergence of [u, v] divided by dt
        d_tmp_v_dy[0, :] = 1 / dx * (-3/2*tmp_v[0, :] + 2*tmp_v[1, :] - 1/2*tmp_v[2, :])
        d_tmp_v_dy[-1, :] = - 1 / dx * (-3/2*tmp_v[N, :] + 2*tmp_v[N - 1, :] - 1/2*tmp_v[N - 2, :])
        d_tmp_v_dy[1:-1, :] = (tmp_v[2:, :] - tmp_v[:-2, :]) / (2*dy)

        d_tmp_u_dx[:, 0] =  1 / dx * (-3/2*tmp_u[:, 0] + 2*tmp_u[:, 1] - 1/2*tmp_u[:, 2])
        d_tmp_u_dx[:, -1] = - 1 / dx * (-3/2*tmp_u[:, N] + 2*tmp_u[:, N - 1] - 1/2*tmp_u[:, N - 2])
        d_tmp_u_dx[:, 1:-1] = (tmp_u[:, 2:] - tmp_u[:, :-2]) / (2*dx)

        p_rhs = (d_tmp_u_dx + d_tmp_v_dy) / dt

        # Step 4.2 gauss seidel iteration to solve for the next
        gauss_step = 1
        while True:
            # sleep(0.01)
            mse = 0.0 # use mean squared error as a convergence criteria
            # laplace p = rhs
            # d_x^2 p + d_y^2 p = rhs
            # update guess and compute mse
            # (2*p[i,j] - 5*p[i,j+1] + 4*p[i,j+2] - p[i,j+3])/dx^2
            # (2*p[i,j] - 5*p[i,j-1] + 4*p[i,j-2] - p[i,j-3])/dx^2
            # (2*p[i,j] - 5*p[i+1,j] + 4*p[i+2,j] - p[i+3,j])/dx^2
            # (2*p[i,j] - 5*p[i-1,j] + 4*p[i-2,j] - p[i-3,j])/dx^2
            # (p[i,j-1] - 2*p[i,j] + p[i,j+1])/dx^2
            # (p[i-1,j] - 2*p[i,j] + p[i+1,j])/dx^2
            # we use jacobi there bc. its easier to implement
            new_p[0,:] = new_p[-1,:] = new_p[:,0] = new_p[:,-1] = p0
            new_p[1:-1, 1:-1] = (p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1]) / 4 - dx**2*p_rhs[1:-1, 1:-1]/4
            mse = np.sum((p - new_p)**2) / (N - 1)**2
            p = (1 - omega) * p + omega * new_p

            if mse < tol:
                break
            if np.isnan(mse) or np.isinf(mse):
                print(p)
                raise ValueError("jacobi iteration diverged")
            if gauss_step > max_steps:
                raise ValueError("jacobi iteration took too long")

            gauss_step += 1

        # Step 5. compute the real velocity using the gradient of the pressure and set boundary conditions
        # Step 5.1 compute the pressure gradient
        grad_p_x = (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx)
        grad_p_y = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy)
        # Step 5.2 advance u and v using forward euler
        u[1:-1, 1:-1] = tmp_u[1:-1, 1:-1] + dt * (- grad_p_x)
        v[1:-1, 1:-1] = tmp_v[1:-1, 1:-1] + dt * (- grad_p_y)

        # Step 6. repeat until tspan is reached
    return u, v


def main():
    N = 60
    Re = 100.0
    U0 = 1
    x = y = np.linspace(0,1,N+1)
    u, v = solve_lid_driven_cavity_flow(N=N, Re=Re, U0=U0)
    # Step 7. visualize the result
    plt.contourf(x,y, np.sqrt(u**2 + v**2), 20)
    plt.colorbar()
    plt.streamplot(x,y,u,v)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lid Driven Cavity Flow, Re = 100, U0 = 1")
    plt.savefig("lid_driven_cavity_flow.pdf")

main()
