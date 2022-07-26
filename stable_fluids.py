import numpy as np, matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy.interpolate as interp
import scipy.integrate as inte

L1 = 1.0
L2 = 1.0
N = 41
M = 42
dt = 0.1
tspan = 10.0
g = 0.0
Re = 1e3
u0 = 1.0
dim = 2

nsteps = int(np.ceil(tspan / dt))

dx = L1 / (N - 1)
dy = L2 / (M - 1)
xs = np.linspace(0, L1, N)
ys = np.linspace(0, L2, M)
yy, xx = np.meshgrid(ys, xs)
rr = np.dstack([xx, yy])

# initial conditions
u = np.zeros((N, M, dim))
u[0, :, 1] = u0

grad_P = np.zeros((N, M, dim))
div_w3 = np.zeros((N, M))

def op_fn_diffusion(v):
    w = v.reshape(N, M, dim)
    laplace = np.zeros((N, M, dim))
    for d in range(dim):
        laplace[1:-1, 1:-1, d] = (
                (w[2:, 1:-1, d] - 2*w[1:-1, 1:-1, d] + w[:-2, 1:-1, d]) / dx**2 +
                (w[1:-1, 2:, d] - 2*w[1:-1, 1:-1, d] + w[1:-1, :-2, d]) / dy**2
        )
    return w - 1 / Re * dt * laplace # I dont need to ravel back?????
op_diff = sla.LinearOperator((N*M*dim, N*M*dim), matvec=op_fn_diffusion)

def op_fn_pressure(v):
    P = v.reshape(N, M)
    laplace = np.zeros((N, M))
    laplace[1:-1, 1:-1] = (
            (P[2:, 1:-1] - 2*P[1:-1, 1:-1] + P[:-2, 1:-1]) / dx**2 +
            (P[1:-1, 2:] - 2*P[1:-1, 1:-1] + P[1:-1, :-2]) / dy**2
    )
    return laplace
op_laplace_pressure = sla.LinearOperator((N*M, N*M), matvec=op_fn_pressure)

for i in range(nsteps):
    print(f"step: {i + 1} of {nsteps}")
    # force
    f = -g
    w1 = u + dt * f

    # advect
    # d a / d t + (v * nabal) a = 0
    # a(x, t) = a(p(x, t), t0)
    # dp/dt (t) = v(t)
    advected_ps = rr - dt * w1
    np.clip(advected_ps[:, :, 0], 0, L1, out=advected_ps[:, :, 0])
    np.clip(advected_ps[:, :, 1], 0, L2, out=advected_ps[:, :, 1])
    w2 = interp.interpn((xs, ys), w1, advected_ps)

    # diffuse
    # (1 - 1/Re * dt * laplace) w3 = w2
    w3_vec, status = sla.cg(op_diff, w2.ravel())
    assert status == 0
    w3 = w3_vec.reshape(N, M, dim)

    # find pressure
    div_w3[1:-1, 1:-1] = (
            (w3[2:, 1:-1, 0] - w3[:-2, 1:-1, 0]) / (2*dx) +
            (w3[1:-1, 2:, 1] - w3[1:-1, :-2, 1]) / (2*dy)
    )
    P_vec, status = sla.cg(op_laplace_pressure, div_w3.ravel())
    assert status == 0
    P = P_vec.reshape(N, M)

    # add pressure
    grad_P[1:-1, 1:-1, 0] = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2*dx)
    grad_P[1:-1, 1:-1, 1] = (P[1:-1, 2:] - P[1:-1, :-2]) / (2*dy)
    u = w3 - dt * grad_P

plt.figure()
plt.streamplot(xs, ys, u[:, :, 0].T, u[:, :, 1].T)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
