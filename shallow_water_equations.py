############ solve the shallow water equations using the MacCormack method ##############

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

####################################### configuration ###################################
# physical parameters in m/s/kg units
Lx = 1e3 * 10 # [m]
Ly = 1e3 * 10 # [m]
H = 4000 # mean height ~ ocean depth [m]
g = 9.81 # acceleration due to gravity on earth (mean) [m/s^2]
b = 0 # viscous drag coefficient. I don't understand were this comes from
# https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html
nu = 0.0000013081 # kinematic viscosity [m^2/s] of water at 10C

# solver parameters
dt = 0.05 # [s]
N = 200; dx = Lx / (N - 1)
Ny = int(Ly / dx) + 1
Nx = int(Lx / dx) + 1
shape = Nx, Ny
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

################################# initial conditions ################################
# velocity in x direction
u0 = np.zeros(shape)
# velocity in y direction
v0 = np.zeros(shape)
# h height derivation
xx, yy = np.meshgrid(x, y)
sharpness = 0.01
R = np.sqrt(Lx**2 + Ly**2)
s = 10
h0 = (
    H / 10 * np.exp(- ((xx - Lx/2 - Lx/s)**2 + (yy - Ly/2 - Ly/s)**2) / (2 * ((R * sharpness)**2)))
    + H / 10 * np.exp(- ((xx - Lx/2 + Lx/s)**2 + (yy - Ly/2 + Ly/s)**2) / (2 * ((R * sharpness)**2)))
)
h0[:, 0] = h0[0, :] = h0[:, -1] = h0[-1, :] = 0.0

################################ derivative operators ######################################
# neumann boundary conditions (I think they are correct for u and v
# but not for h). At least h can change along the bounary.
# This would also impy different bounary conditions for A_u and A_v.
# storage for the derivatives
dh_dx   = np.zeros(shape)
dh_dy   = np.zeros(shape)
du_dx   = np.zeros(shape)
du_dy   = np.zeros(shape)
dv_dx   = np.zeros(shape)
dv_dy   = np.zeros(shape)
d2u_dx2 = np.zeros(shape)
d2u_dy2 = np.zeros(shape)
d2v_dx2 = np.zeros(shape)
d2v_dy2 = np.zeros(shape)

def diff(left, right):
    return (left - right) / dx

def diffx_forward(field):
    c = field[:, 1:-1]
    r = field[:, 2:]
    return diff(c, r)

def diffy_forward(field):
    c = field[1:-1, :]
    r = field[2:,   :]
    return diff(c, r)

def diffx_backward(field):
    l = field[:, :-2]
    c = field[:, 1:-1]
    return diff(l, c)

def diffy_backward(field):
    l = field[:-2,  :]
    c = field[1:-1, :]
    return diff(l, c)

# second derivative
def diff2(left, center, right):
    return (left - 2 * center + right) / dx**2

def diff2x(field):
    l = field[1:-1, :-2]
    c = field[1:-1, 1:-1]
    r = field[1:-1, 2:]
    return diff2(l, c, r)

def diff2y(field):
    l = field[:-2,  1:-1]
    c = field[1:-1, 1:-1]
    r = field[2:,   1:-1]
    return diff2(l, c, r)

############################ integration of the eom ##############################
def rhs(h, u, v, is_predictor):
    # compute all spacial derivatives by finte central differences
    if is_predictor:
        dh_dx[:, 1:-1] = diffx_forward(h)
        dh_dy[1:-1, :] = diffy_forward(h)
        du_dx[1:-1, 1:-1] = diffx_forward(u)[1:-1, :]
        du_dy[1:-1, 1:-1] = diffy_forward(u)[:, 1:-1]
        dv_dx[1:-1, 1:-1] = diffx_forward(v)[1:-1, :]
        dv_dy[1:-1, 1:-1] = diffy_forward(v)[:, 1:-1]
    else:
        dh_dx[:, 1:-1] = diffx_backward(h)
        dh_dy[1:-1, :] = diffy_backward(h)
        du_dx[1:-1, 1:-1] = diffx_backward(u)[1:-1, :]
        du_dy[1:-1, 1:-1] = diffy_backward(u)[:, 1:-1]
        dv_dx[1:-1, 1:-1] = diffx_backward(v)[1:-1, :]
        dv_dy[1:-1, 1:-1] = diffy_backward(v)[:, 1:-1]

    d2u_dx2[1:-1, 1:-1] = diff2x(u)
    d2u_dy2[1:-1, 1:-1] = diff2y(u)
    d2v_dx2[1:-1, 1:-1] = diff2x(v)
    d2v_dy2[1:-1, 1:-1] = diff2y(v)
    laplace_u = d2u_dx2 + d2u_dy2
    laplace_v = d2v_dx2 + d2v_dy2

    # compute rhs
    # assume H = const for now
    dh_dt = - dh_dx * u - (h + H) * du_dx - dh_dy * v - (h + H) * dv_dy
    du_dt = - u * du_dx - v * du_dy - g * dh_dx - b * u + nu * laplace_u
    dv_dt = - u * dv_dx - v * dv_dy - g * dh_dy - b * v + nu * laplace_v

    return dh_dt, du_dt, dv_dt

def do_step(h, u, v):
    # MacCormack method:
    # predictor step using forward differences
    dh_dt, du_dt, dv_dt = rhs(h, u, v, True)
    h_pred = h + dt * dh_dt
    u_pred = u + dt * du_dt
    v_pred = v + dt * dv_dt

    # corrector step using backward differences at the half step
    h_half_step = (h + h_pred) / 2.0
    u_half_step = (u + u_pred) / 2.0
    v_half_step = (v + v_pred) / 2.0
    dh_dt_pred, du_dt_pred, dv_dt_pred = rhs(h_pred, u_pred, v_pred, False)

    h_corr = h_half_step + dt / 2.0 * dh_dt_pred
    u_corr = u_half_step + dt / 2.0 * du_dt_pred
    v_corr = v_half_step + dt / 2.0 * dv_dt_pred

    return h_corr, u_corr, v_corr


######################## main simulation and display loop #########################
t = 0
current_h, current_u, current_v = h0, u0, v0
while True:
    plt.clf()
    span = np.max(np.abs(current_h))
    norm = mpl.colors.Normalize(vmin=-span, vmax=+span)
    km = 1e3
    plt.pcolormesh(x / km, y / km, current_h, norm=norm, cmap="RdBu_r")
    plt.colorbar(norm=norm).set_label("derivation around the mean, h / m")
    plt.xlabel("x / km")
    plt.ylabel("y / km")
    plt.title(f"t = {t:.2f} s")
    plt.pause(0.001)
    start = time.time()
    current_h, current_u, current_v = do_step(current_h, current_u, current_v)
    end = time.time()
    print("time for step:", end - start)
    t += dt

