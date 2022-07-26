import numpy as np, matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy.interpolate as interp
import scipy.integrate as inte

plt.ion()

L1 = 10.0
L2 = 1.0
N = 50
M = 50
dt = 2.0
tspan = 100.0
Re = 1e3
u0 = 1.0
dim = 2
sphere_x = 3.0
sphere_y = L2/2
sphere_r = 0.1

nsteps = int(np.ceil(tspan / dt))

dx = L1 / (N - 1)
dy = L2 / (M - 1)
xs = np.linspace(0, L1, N)
ys = np.linspace(0, L2, M)
yy, xx = np.meshgrid(ys, xs)
rr = np.dstack([xx, yy])

# initial conditions
u = np.zeros((N, M, dim))
u[0, :, 0] = u0
u[-1, :, 0] = u0

is_boundary = (
        (xx == 0) | (xx == L1) | (yy == 0) | (yy == L2) |
        ((xx - sphere_x)**2 + (yy - sphere_y)**2 < sphere_r**2)
)

grad_P = np.zeros((N, M, dim))
div_w3 = np.zeros((N, M))
P_vec = w3_vec = None

def op_fn_diffusion(v):
    w = v.reshape(N, M, dim)
    laplace = np.zeros((N, M, dim))
    for d in range(dim):
        laplace[1:-1, 1:-1, d] = (
                (w[2:, 1:-1, d] - 2*w[1:-1, 1:-1, d] + w[:-2, 1:-1, d]) / dx**2 +
                (w[1:-1, 2:, d] - 2*w[1:-1, 1:-1, d] + w[1:-1, :-2, d]) / dy**2
        )
    for i in range(dim): np.where(is_boundary, 0.0, laplace[..., i])
    return w - 1 / Re * dt * laplace
op_diff = sla.LinearOperator((N*M*dim, N*M*dim), matvec=op_fn_diffusion)

# on the walls including the sphere: v = 0 and grad p * n = 0
#[p(x - h) - 2*p(x) + p(x + h)] / dx^2
#[p(x + h) - p(x - h)] / (2dx) = 0 ==> p(x - h) = p(x + h)
# 2 * (p(x + h) - p(x)) / dx^2

# on the in and outlets: grad v * n = 0, p = const (inflow = 1.0, outflow = -1.0)

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
    w2[is_boundary, :] = u[is_boundary, :]

    # diffuse
    # (1 - 1/Re * dt * laplace) w3 = w2
    w3_vec, status = sla.cg(op_diff, w2.ravel(), x0=w3_vec)
    assert status == 0
    w3 = w3_vec.reshape(N, M, dim)

    # find pressure
    # TODO: some of these terms are zero (derivative tangential along the walls of the pipe)
    div_w3[ 0,  0] = (w3[+1,  0, 0] - w3[ 0,  0, 0]) / dx + (w3[ 0, +1, 1] - w3[ 0,  0, 1]) / dy
    div_w3[ 0, -1] = (w3[+1, -1, 0] - w3[ 0, -1, 0]) / dx + (w3[ 0, -1, 1] - w3[ 0, -2, 1]) / dy
    div_w3[-1,  0] = (w3[-1,  0, 0] - w3[-2,  0, 0]) / dx + (w3[-2, +1, 1] - w3[-1,  0, 1]) / dy
    div_w3[-1, -1] = (w3[-1, -1, 0] - w3[-2, -1, 0]) / dx + (w3[-1, -1, 1] - w3[-1, -2, 1]) / dy
    div_w3[ 0, 1:-1] = (w3[+1, 1:-1, 0] - w3[ 0, 1:-1, 0]) / dx +     (w3[ 0,   2:, 1] - w3[ 0,  :-2, 1]) / (2*dy)
    div_w3[-1, 1:-1] = (w3[-1, 1:-1, 0] - w3[-2, 1:-1, 0]) / dx +     (w3[-1,   2:, 1] - w3[-1,  :-2, 1]) / (2*dy)
    div_w3[1:-1,  0] = (w3[2:, 0   , 0] - w3[:-2,  0 , 0]) / (2*dx) + (w3[1:-1, +1, 1] - w3[1:-1,  0, 1]) / dy
    div_w3[1:-1, -1] = (w3[2:, -1  , 0] - w3[:-2, -1 , 0]) / (2*dx) + (w3[1:-1, -1, 1] - w3[1:-1, -2, 1]) / dy
    div_w3[1:-1, 1:-1] = (
            (w3[2:, 1:-1, 0] - w3[:-2, 1:-1, 0]) / (2*dx) +
            (w3[1:-1, 2:, 1] - w3[1:-1, :-2, 1]) / (2*dy)
    )
    P_vec, status = sla.cg(op_laplace_pressure, div_w3.ravel(), x0=P_vec)
    assert status == 0
    P = P_vec.reshape(N, M)

    # add pressure
    grad_P[1:-1, 1:-1, 0] = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2*dx)
    grad_P[1:-1, 1:-1, 1] = (P[1:-1, 2:] - P[1:-1, :-2]) / (2*dy)
    w4 = w3 - dt * grad_P
    w4[is_boundary, :] = u[is_boundary, :]
    u = w4

def plot(xs, ys, u, contourf=True, colorbar=True, streamplot=True):
    plt.figure()
    plt.axes().set_aspect("equal")
    if contourf: plt.contourf(xs, ys, np.linalg.norm(u, axis=2).T)
    if streamplot: plt.streamplot(xs, ys, u[:, :, 0].T, u[:, :, 1].T)
    if colorbar: plt.colorbar(label="|v|", orientation="horizontal")
    plt.xlabel("x")
    plt.ylabel("y")
