import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg

np.random.seed(20132)

# physics
Lx = 1.0
Ly = 1.0
tspan = 1.0
T_top = 0.0
T_bottom = 100.0
nu = 10 # viscosity
kappa = 0.01 # thermal conductivity
beta = -100.0 # thermal expandsion coefficient * acceleartion due to gravity

# numerics
epsilon = 1e-10
omega = 0.3
Nx = 20
Ny = 20
dt = 0.01

nsteps = int(tspan / dt)
dx = Lx / Nx
dy = Ly / (Ny - 1)
xs = np.linspace(0, Lx, Nx)
ys = np.linspace(0, Ly, Ny)
v = np.zeros((Nx, Ny, 2))
v[1:-1, 1:-1] = np.random.randn(Nx - 2, Ny - 2, 2) * 0.01
v_advec_bouyancy_x_old = v_advec_bouyancy_y_old = T_advec_old = None
P = np.random.randn(Nx, Ny - 1)
T = np.zeros((Nx, Ny))
Ty = (T_bottom - T_top) * ys / Ly + T_top
T0 = (T_bottom + T_top) / 2
T[:, 0] = T_top
T[:, -1] = T_bottom
T[:, 1:-1] = np.random.randn(Nx, Ny - 2) * 0.1 + Ty[None, 1:-1]

def left(field): return np.roll(field, 1, axis=0)
def right(field): return np.roll(field, -1, axis=0)

def solve_implicit_euler(field0, D):
    # dq/dt = D*laplsce q
    # q(t + 1) = q(t) + dt * D * laplace q(t + 1)
    # q(t) = q(t + 1) - dt * D * laplace q(t + 1)
    def apply_rhs_implicit_euler(field):
        field = field.reshape((Nx, Ny))
        laplace = (left(field[:, 1:-1]) - 2 * field[:, 1:-1] + right(field[:, 1:-1])) / dx**2 + (field[:, :-2] - 2*field[:, 1:-1] + field[:, 2:]) / dy**2
        interior = field[:, 1:-1] - dt * D * laplace
        return np.hstack([field[:, 0:1], interior, field[:, -1:]]).reshape(-1)

    op = LinearOperator(shape=(Nx * Ny, Nx * Ny), matvec=apply_rhs_implicit_euler)
    field, status = cg(op, field0.reshape(-1), x0=field0.reshape(-1), tol=epsilon)
    assert status == 0
    return field.reshape((Nx, Ny))

def apply_laplace_pressure(P):
    P = P.reshape((Nx, Ny - 1))
    interior = (left(P[:, 1:-1]) - 2 * P[:, 1:-1] + right(P[:, 1:-1])) / dx**2 + (P[:, :-2] - 2*P[:, 1:-1] + P[:, 2:]) / dy**2
    top = (left(P[:, 0]) - 2 * P[:, 0] + right(P[:, 0])) / dx**2 + (- P[:, 0] + P[:, 1]) / dy**2
    bottom = (left(P[:, -1]) - 2 * P[:, -1] + right(P[:, -1])) / dx**2 + (- P[:, -1] + P[:, -2]) / dy**2
    laplace = np.hstack([top[:, None], interior, bottom[:, None]])
    return laplace.reshape(-1)

def solve_pressure_poisson(div_v, P):
    d = Nx * (Ny - 1)
    op = LinearOperator(shape=(d,d), matvec=apply_laplace_pressure)
    P, status = cg(op, div_v.reshape(-1), x0=P.reshape(-1), tol=epsilon)
    assert status == 0
    return P.reshape((Nx, Ny - 1))

for step in range(nsteps):
    print(f"step = {step + 1} / {nsteps}")
    ###################### advection #####################
    # dv/dt = - (v dot nabal) v
    # upwind difference (take the signle sided finite diff in upwind direction)
    dvxdx = np.where(v[:, 1:-1, 0] < 0.0, right(v)[:, 1:-1, 0] - v[:, 1:-1, 0], v[:, 1:-1, 0] - left(v)[:, 1:-1, 0]) / dx
    dvydx = np.where(v[:, 1:-1, 0] < 0.0, right(v)[:, 1:-1, 1] - v[:, 1:-1, 1], v[:, 1:-1, 1] - left(v)[:, 1:-1, 1]) / dx
    dvxdy = np.where(v[:, 1:-1, 1] < 0.0, v[:, 2:, 0] - v[:, 1:-1, 0], v[:, 1:-1, 0] - v[:, :-2, 0]) / dy
    dvydy = np.where(v[:, 1:-1, 1] < 0.0, v[:, 2:, 1] - v[:, 1:-1, 1], v[:, 1:-1, 1] - v[:, :-2, 1]) / dy
    v_advec_bouyancy_x = - (v[:, 1:-1, 0] * dvxdx + v[:, 1:-1, 1] * dvxdy)
    v_advec_bouyancy_y = - (v[:, 1:-1, 0] * dvydx + v[:, 1:-1, 1] * dvydy)
    dTdx = np.where(v[:, 1:-1, 0] < 0.0, right(T)[:, 1:-1] - T[:, 1:-1], T[:, 1:-1] - left(T)[:, 1:-1]) / dx
    dTdy = np.where(v[:, 1:-1, 1] < 0.0, T[:, 2:] - T[:, 1:-1], T[:, 1:-1] - T[:, :-2]) / dy
    T_advec = - (v[:, 1:-1, 0] * dTdx + v[:, 1:-1, 1] * dTdy)
    v_advec_bouyancy_y += beta * (T[:, 1:-1] - T0)
    if v_advec_bouyancy_x_old is None:
        # forward euler
        v[:, 1:-1, 0] += dt * v_advec_bouyancy_x
        v[:, 1:-1, 1] += dt * v_advec_bouyancy_y
        T[:, 1:-1] += dt * T_advec
    else:
        # adams-bashford 2nd order
        v[:, 1:-1, 0] += dt * (3 * v_advec_bouyancy_x - v_advec_bouyancy_x_old) / 2
        v[:, 1:-1, 1] += dt * (3 * v_advec_bouyancy_y - v_advec_bouyancy_y_old) / 2
        T[:, 1:-1] += dt * (3 * T_advec - T_advec_old) / 2
        v_advec_bouyancy_x_old = v_advec_bouyancy_x
        v_advec_bouyancy_y_old = v_advec_bouyancy_y
        T_advec_old = T_advec

    ####################### diffusion #######################
    # implicit euler
    # d v / dt = D * laplace v
    v[:, :, 0] = solve_implicit_euler(v[:, :, 0], nu)
    v[:, :, 1] = solve_implicit_euler(v[:, :, 1], nu)
    T = solve_implicit_euler(T, kappa)

    ######################### pressure #########################
    dvxdx = (right(v)[:, :, 0] - left(v)[:, :, 0]) / dx
    dvydy = (v[:, 1:, 1] - v[:, :-1, 1]) / dy
    div_v = (dvxdx[:, :-1] + dvxdx[:, 1:]) / 2 + (left(dvydy) + right(dvydy)) / 2
    div_v /= dt
    P = solve_pressure_poisson(div_v, P)
    dPdx = (right(P) - left(P)) / dx
    dPdy = (P[:, 1:] - P[:, :-1]) / dy
    v[:, 1:-1, 0] -= dt * (dPdx[:, 1:] + dPdx[:, :-1]) / 2
    v[:, 1:-1, 1] -= dt * (left(dPdy) + right(dPdy)) / 2

    plt.clf()
    plt.subplot(2,1,1)
    plt.contourf(xs, ys, np.linalg.norm(v, axis=2).T)
    plt.colorbar(label="|v| / a.u.")
    plt.quiver(xs, ys, v[:, :, 0].T, v[:, :, 1].T)
    plt.gca().get_xaxis().set_visible(False)
    plt.xlabel("x / a.u.")
    plt.ylabel("y / a.u.")
    plt.subplot(2,1,2)
    plt.contourf(xs, ys, T.T)
    plt.xlabel("x / a.u.")
    plt.ylabel("y / a.u.")
    plt.colorbar(label="T / a.u.")
    plt.suptitle("Convection Cell")
    plt.pause(0.001)

plt.show()
