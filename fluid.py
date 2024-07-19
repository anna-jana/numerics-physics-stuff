import numpy as np
import matplotlib.pyplot as plt
from numba import njit

plt.ion()
np.random.seed(20132)

def residual(r, grid, dx, dy, rhs):
    (left(grid[1:-1]) - 2*grid[:, 1:-1] + right(grid[:, 1:-1])) / dx**2 + (grid[:, :-1] - 2 * grid[:, 1:-1] + grid[:, 2:]) / dy**2

# physics
Lx = 1.0
Ly = 1.0
tspan = 0.1
T_top = 0.0
T_bottom = 10.0
nu = 0.01 # viscosity
kappa = 1.0 # thermal conductivity
beta = -1.0 # thermal expandsion coefficient * acceleartion due to gravity

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

def solve_implicit_euler(init, D, name):
    # dq/dt = D*laplsce q
    # q(t + 1) = q(t) + dt * D * laplace q(t + 1)
    # q(t) = q(t + 1) - dt * D * laplace q(t + 1)
    # q(t) = q(t + 1) - dt * D * (laplace_off_diag q(t + 1) - (1/dx**2 + 1/dy**2) * 2 * q(t + 1))
    # q(t) = q(t + 1) * (1 + dt * D * (1/dx**2 + 1/dy**2) * 2) - D * dt * laplace_off_diag q(t + 1)
    prev = init
    new = init.copy()
    steps = 0
    while True:
        off_diag_laplace = (left(prev[:, 1:-1]) + right(prev[:, 1:-1])) / dx**2 + (prev[:, 2:] + prev[:, :-2]) / dy**2
        diag = 1 + dt * D * 2 * (1/dx**2 + 1/dy**2)
        new[:, 1:-1] = (init[:, 1:-1] - dt * D * off_diag_laplace) / diag
        new = omega * new + (1 - omega) * prev
        mse = np.mean(((prev - new) / (1 - omega))**2)
        prev, new = new, prev
        if not np.isfinite(mse):
            raise ValueError("implicit euler diverged")
        if mse < epsilon:
            print(f"implicit euler for {name} diffusion:", steps, "steps")
            return new
        steps += 1

def solve_pressure_poisson(div_v, P):
    new_P = P.copy()
    steps = 0
    while True:
        # laplace P = div v
        # (L - 2*C + R) / dx^2 + (T - 2*C + B) / dy^2 = div v
        # (L + R) / dx^2 + (T + B) / dy^2 - 2 * C * (1/dx^2 + 1/dy^2) = div v
        # (L + R) / dx^2 + (T + B) / dy^2 - div v = 2 * C * (1/dx^2 + 1/dy^2)
        diag = 2 * (1/dx**2 + 1/dy**2)
        laplace_off_diag = (left(P[:, 1:-1]) + right(P[:, 1:-1])) / dx**2 + (P[:, 2:] + P[:, :-2]) / dy**2
        new_P[:, 1:-1] = (- div_v[:, 1:-1] + laplace_off_diag) / diag
        # @ y = 0: grad P = 0 ==> T - C = 0
        # (L + R) / dx^2 + (C + B) / dy^2 - div v = 2 * C * (1/dx^2 + 1/dy^2)
        # (L + R) / dx^2 + B / dy^2 - div v = 2 * C * (1/dx^2 + 1/dy^2) - C / dy^2 = C * (2/dx^2 + 1/dy^2)
        laplace_off_diag = (left(P[:, 0]) + right(P[:, 0])) / dx**2 + P[:, 1] / dy**2
        diag = 2/dx**2 + 1/dy**2
        new_P[:, 0] = (- div_v [:, 0] + laplace_off_diag) / diag

        laplace_off_diag = (left(P[:, -1]) + right(P[:, -1])) / dx**2 + P[:, -2] / dy**2
        new_P[:, -1] = (- div_v[:, -1] + laplace_off_diag) / diag # reuse diag

        new_P = (1 - omega) * P + omega * new_P
        # new_P - P = (1 - omega) * new_P + omega * P - P = (1 - omega) * new_P + (omega - 1) * P
        # = (1 - omega) * (new_P - P)
        mse = np.mean(((P - new_P) / (1 - omega))**2)
        if not np.isfinite(mse):
            raise ValueError("pressure solver diverged")
        P, new_P = new_P, P
        if mse < epsilon:
            print("pressure:", steps, "steps")
            return P
        steps += 1

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
    v = solve_implicit_euler(v, nu, "viscosity")
    T = solve_implicit_euler(T, kappa, "thermal conduction")

    ######################### pressure #########################
    dvxdx = (right(v)[:, :, 0] - left(v)[:, :, 0]) / dx
    dvydy = (v[:, 1:, 1] - v[:, :-1, 1]) / dy
    div_v = (dvxdx[:, :-1] + dvxdx[:, 1:]) / 2 + (left(dvydy) + right(dvydy)) / 2
    div_v /= dt
    P = solve_pressure_poisson(div_v, P)
    dPdx = (right(P) - left(P)) / dx
    dPdy = (P[:, 1:] - P[:, :-1]) / dy
    dPdx = (dPdx[:, 1:] + dPdx[:, :-1]) / 2
    dPdy = (left(dPdy) + right(dPdy)) / 2
    v[:, 1:-1, 0] -= dt * dPdx
    v[:, 1:-1, 1] -= dt * dPdy

    dvxdx = (right(v)[:, :, 0] - left(v)[:, :, 0]) / dx
    dvydy = (v[:, 1:, 1] - v[:, :-1, 1]) / dy
    div_v = (dvxdx[:, :-1] + dvxdx[:, 1:]) / 2 + (left(dvydy) + right(dvydy)) / 2
    print("div v:", max(Lx, Ly) * np.sqrt(np.mean(div_v**2)) / np.sqrt(np.mean(v**2)))

plt.figure()
plt.subplot(3,1,1)
plt.contourf(xs, ys, np.linalg.norm(v, axis=2).T)
plt.colorbar(label="|v| / a.u.")
plt.quiver(xs, ys, v[:, :, 0].T, v[:, :, 1].T)
plt.gca().invert_yaxis()
plt.xlabel("x / a.u.")
plt.ylabel("y / a.u.")
plt.subplot(3,1,2)
plt.contourf(xs, ys, T.T)
plt.xlabel("x / a.u.")
plt.ylabel("y / a.u.")
plt.gca().invert_yaxis()
plt.colorbar(label="T / a.u.")
plt.subplot(3,1,3)
plt.contourf(xs, ys[:-1] + dy/2, P.T)
plt.xlabel("x / a.u.")
plt.ylabel("y / a.u.")
plt.gca().invert_yaxis()
plt.colorbar(label="P / a.u.")
plt.show()
