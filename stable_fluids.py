import time
import numpy as np, matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy.interpolate as interp

# dv/dt + (v * nabal) v = grad P - 1 / Re * laplace v
# laplace p = div v
# p_inlet = p_inflow
# p_outlet = p_outflow
# on the walls including the box: v = 0 and grad p * n = 0
# stable fluids method: https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf

np.random.seed(42)

L1 = 10.0
L2 = 1.0
h = 0.1
N = int(0.5 + L1 / h)
M = int(0.5 + L2 / h)
dt = 2.0
tspan = 100.0
Re = 1e3
P_inflow = 1.0
P_outflow = -1.0
dim = 2
box_h = box_w = 0.3
box_x = L1 / 10 * 3 - box_w / 2
box_y = L2 / 2 - box_h / 2

nsteps = int(np.ceil(tspan / dt))

dx = L1 / (N - 1)
dy = L2 / (M - 1)
xs = np.linspace(0, L1, N)
ys = np.linspace(0, L2, M)
yy, xx = np.meshgrid(ys, xs)
rr = np.dstack([xx, yy])

box_start_x = int(np.ceil(box_x / dx))
box_end_x = int(np.ceil((box_x + box_w) / dx))
box_start_y = int(np.ceil(box_y / dy))
box_end_y = int(np.ceil((box_y + box_h) / dy))

is_box = np.zeros((N, M), dtype="bool")
is_box[box_start_x:box_end_x, box_start_y:box_end_y] = True
is_boundary = (xx == 0) | (xx == L1) | (yy == 0) | (yy == L2) | is_box

grad_P = np.zeros((N, M, dim))
rhs_pressure = np.zeros((N, M))
P_vec = np.random.randn(N*M)
w3_vec = None
u = np.zeros((N, M, dim))

def get_matrix(op):
    n = op.shape[0]
    mat = np.empty(op.shape)
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1
        mat[:, i] = op(v)
    return mat

def plot(xs, ys, u, P, contourf=True, streamplot=True):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    # axs[0].set_aspect("equal")
    if contourf:
        cont = axs[0].contourf(xs, ys, np.linalg.norm(u, axis=2).T)
        fig.colorbar(cont, ax=axs[0], label="|v|")
    if streamplot:
        axs[0].streamplot(xs, ys, u[:, :, 0].T, u[:, :, 1].T)
    axs[0].fill([box_x, box_x, box_x + box_w, box_x + box_w],
                [box_y, box_y + box_h, box_y + box_h, box_y],
                "white", zorder=100)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    # axs[1].set_aspect("equal")
    cont = axs[1].contourf(xs, ys, P.T)
    fig.colorbar(cont, ax=axs[1], label="P")
    axs[1].fill([box_x, box_x, box_x + box_w, box_x + box_w],
                [box_y, box_y + box_h, box_y + box_h, box_y],
                "white")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

def op_fn_diffusion(v):
    w = v.reshape(N, M, dim)
    # dont change the velocity on the walls:
    laplace = np.zeros((N, M, dim))
    # iterior:
    laplace[1:-1, 1:-1, :] = (
            (w[2:, 1:-1, :] - 2*w[1:-1, 1:-1, :] + w[:-2, 1:-1, :]) / dx**2 +
            (w[1:-1, 2:, :] - 2*w[1:-1, 1:-1, :] + w[1:-1, :-2, :]) / dy**2
    )
    # dont change the velocity on the sphere:
    for i in range(dim): laplace[is_box, i] = 0.0
    # inlet and outlet:
    # grad v * n = 0
    # (dv/dx, dv/dy) * (1, 0) = dv/dx = 0 for both vx and vy
    laplace[0, 1:-1, :] = (
            2 * (w[1, 1:-1, :] - w[0, 1:-1, :]) / dx**2 +
            (w[0, 2:, :] - 2*w[0, 1:-1, :] + w[0, :-2, :]) / dy**2
    )
    laplace[-1, 1:-1, :] = (
            2 * (w[-1, 1:-1, :] - w[-2, 1:-1, :]) / dx**2 +
            (w[-1, 2:, :] - 2*w[-1, 1:-1, :] + w[-1, :-2, :]) / dy**2
    )
    return w - 1 / Re * dt * laplace
op_diff = sla.LinearOperator((N*M*dim, N*M*dim), matvec=op_fn_diffusion)

def op_fn_pressure(v):
    P = v.reshape(N, M)
    lhs_pressure = np.zeros((N, M))

    # interiour laplace P = div v
    # multiply with dx^2 to make the matrix better conditioned (|det| smaller)
    lhs_pressure[1:-1, 1:-1] = (
            (P[2:, 1:-1] - 2*P[1:-1, 1:-1] + P[:-2, 1:-1]) +
            (P[1:-1, 2:] - 2*P[1:-1, 1:-1] + P[1:-1, :-2]) * dx**2 / dy**2
    )

    # inflow and outflow:
    lhs_pressure[0, :] = P[0, :]
    lhs_pressure[-1, :] = P[-1, :]

    # walls:
    lhs_pressure[1:-1, 0] = P[1:-1, 1] - P[1:-1, 0]
    lhs_pressure[1:-1, -1] = P[1:-1, -1] - P[1:-1, -2]

    # box
    lhs_pressure[box_start_x+1:box_end_x-1, box_start_y] = (
            P[box_start_x+1:box_end_x-1, box_start_y    ] -
            P[box_start_x+1:box_end_x-1, box_start_y + 1]
    )
    lhs_pressure[box_start_x+1:box_end_x-1, box_end_y] = (
            P[box_start_x+1:box_end_x-1, box_end_y - 1] -
            P[box_start_x+1:box_end_x-1, box_end_y    ]
    )
    lhs_pressure[box_start_x, box_start_y+1:box_end_y-1] = (
            P[box_start_x    , box_start_y+1:box_end_y-1] -
            P[box_start_x + 1, box_start_y+1:box_end_y-1]
    )
    lhs_pressure[box_end_x, box_start_y+1:box_end_y-1] = (
            P[box_end_x - 1, box_start_y+1:box_end_y-1] -
            P[box_end_x    , box_start_y+1:box_end_y-1]
    )

    # box: +: normal interiour point, *: point where grad P * n = 0, .: point within the box
    # +***************+
    # *...............*
    # *...............*
    # *...............*
    # *...............*
    # +***************+

    lhs_pressure[box_start_x+1:box_end_x-1, box_start_y+1:box_end_y-1] = \
            P[box_start_x+1:box_end_x-1, box_start_y+1:box_end_y-1]

    return lhs_pressure
op_laplace_pressure = sla.LinearOperator((N*M, N*M), matvec=op_fn_pressure)

begin = time.perf_counter_ns()

for i in range(nsteps):
    print(f"step: {i + 1} of {nsteps} ...", end="", flush=True)
    start = time.perf_counter_ns()

    # force
    f = 0.0 # no force implemented
    w1 = u + dt * f

    # advect
    # d a / d t + (v * nabal) a = 0
    # a(x, t) = a(p(x, t), t0)
    # dp/dt (t) = v(t)
    advected_ps = rr - dt * w1
    np.clip(advected_ps[:, :, 0], 0, L1, out=advected_ps[:, :, 0])
    np.clip(advected_ps[:, :, 1], 0, L2, out=advected_ps[:, :, 1])
    w2 = interp.interpn((xs, ys), w1, advected_ps)
    w2[:, 0, 0] = w2[:, 0, 1] = w2[:, -1, 0] = w2[:, -1, 1] = 0.0 # v = 0 on the walls
    w2[is_box, 0] = w2[is_box, 1] = 0.0 # v = 0 on the box

    # diffuse
    # (1 - 1/Re * dt * laplace) w3 = w2
    w3_vec, status = sla.cg(op_diff, w2.ravel(), x0=w3_vec)
    assert status == 0
    w3 = w3_vec.reshape(N, M, dim)

    # find pressure
    # laplace P = div v
    # inlet and outlet:
    rhs_pressure[0, :] = P_inflow
    rhs_pressure[-1, :] = P_outflow

    # interiour laplace P = div v
    # multiply with dx^2 to make the matrix better conditioned (|det| smaller)
    rhs_pressure[1:-1, 1:-1] = (
            (w3[2:,   1:-1, 0] - w3[ :-2, 1:-1, 0]) * dx / 2 +
            (w3[1:-1, 2:,   1] - w3[1:-1,  :-2, 1]) * dx**2 / (2*dy)
    )

    # box
    rhs_pressure[box_start_x+1:box_end_x-1, box_start_y] = 0
    rhs_pressure[box_start_x+1:box_end_x-1, box_end_y] = 0
    rhs_pressure[box_start_x, box_start_y+1:box_end_y-1] = 0
    rhs_pressure[box_end_x, box_start_y+1:box_end_y-1] = 0
    rhs_pressure[box_start_x+1:box_end_x-1, box_start_y+1:box_end_y-1] = 0

    # solve system
    P_vec, status = sla.gcrotmk(op_laplace_pressure, rhs_pressure.ravel(), x0=P_vec, atol=1e-4, maxiter=10**3)
    assert status == 0
    P = P_vec.reshape(N, M)

    # add pressure
    grad_P[1:-1, 1:-1, 0] = (P[ 2:,   1:-1] - P[  :-2, 1:-1]) / (2*dx)
    grad_P[1:-1, 1:-1, 1] = (P[ 1:-1, 2:  ] - P[ 1:-1,  :-2]) / (2*dy)
    grad_P[ 0,   1:-1, 1] = (P[ 1,    1:-1] - P[ 0,    1:-1]) / dx
    grad_P[ 0,   1:-1, 0] = (P[ 0,     :-2] - P[ 0,    2:  ]) / (2*dy)
    grad_P[-1,   1:-1, 1] = (P[-1,    1:-1] - P[-2,    1:-1]) / dx
    grad_P[-1,   1:-1, 0] = (P[-1,     :-2] - P[-1,    2:  ]) / (2*dy)
    grad_P[box_start_x:box_end_x, box_start_y:box_end_y, :] = 0.0 # boundary cond. on box v = 0

    w4 = w3 - dt * grad_P
    u = w4

    end = time.perf_counter_ns()
    print(f" took {(end - start) / 1e9} seconds")

print(f"total: {(end - begin) / 1e9} seconds")

plot(xs, ys, u, P)
plt.show()
