# solve the shallow water equations using the MacCormack method

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# physical parameters in m/s/kg
Lx = 1e3 * 10 # [m]
Ly = 1e3 * 10 # [m]
H = 4000 # mean height [m]
g = 9.81 # acceleration due to gravity [m/s^2]
b = 0 # viscous drag coefficient
# https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html
nu = 0.0000013081 # kinematic viscosity [m^2/s]

# solver parameters
dt = 0.1
N = 100; dx = Lx / (N - 1)
Ny = int(Ly / dx) + 1
Nx = int(Lx / dx) + 1
shape = Nx, Ny
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# storeage for the dynamical fields
# velocity in x direction
u0 = np.zeros(shape)
# velocity in y direction
v0 = np.zeros(shape)
# h height derivation
xx, yy = np.meshgrid(x, y)
sharpness = 0.05
R = np.sqrt(Lx**2 + Ly**2)
h0 = H / 10 * np.exp(- ((xx - Lx/2)**2 + (yy - Ly/2)**2) / (2 * ((R * sharpness)**2)))
h0[:, 0] = h0[0, :] = h0[:, -1] = h0[-1, :] = 0.0

# storage for the derivatives
dAu_dx  = np.empty(shape)
dAv_dy  = np.empty(shape)
dh_dx   = np.empty(shape)
dh_dy   = np.empty(shape)
du_dx   = np.empty(shape)
du_dy   = np.empty(shape)
dv_dx   = np.empty(shape)
dv_dy   = np.empty(shape)
d2u_dx2 = np.empty(shape)
d2u_dy2 = np.empty(shape)
d2v_dx2 = np.empty(shape)
d2v_dy2 = np.empty(shape)

# neumann boundary conditions
dAu_dx [:, 0] = dAu_dx [:, -1] = dAu_dx  [0, :] = dAu_dx  [-1, :] = 0.0
dAv_dy [:, 0] = dAv_dy [:, -1] = dAv_dy  [0, :] = dAv_dy  [-1, :] = 0.0
dh_dx  [:, 0] = dh_dx  [:, -1] = dh_dx   [0, :] = dh_dx   [-1, :] = 0.0
dh_dy  [:, 0] = dh_dy  [:, -1] = dh_dy   [0, :] = dh_dy   [-1, :] = 0.0
du_dx  [:, 0] = du_dx  [:, -1] = du_dx   [0, :] = du_dx   [-1, :] = 0.0
du_dy  [:, 0] = du_dy  [:, -1] = du_dy   [0, :] = du_dy   [-1, :] = 0.0
dv_dx  [:, 0] = dv_dx  [:, -1] = dv_dx   [0, :] = dv_dx   [-1, :] = 0.0
dv_dy  [:, 0] = dv_dy  [:, -1] = dv_dy   [0, :] = dv_dy   [-1, :] = 0.0
d2u_dx2[:, 0] = d2u_dx2[:, -1] = d2u_dx2 [0, :] = d2u_dx2 [-1, :] = 0.0
d2u_dy2[:, 0] = d2u_dy2[:, -1] = d2u_dy2 [0, :] = d2u_dy2 [-1, :] = 0.0
d2v_dx2[:, 0] = d2v_dx2[:, -1] = d2v_dx2 [0, :] = d2v_dx2 [-1, :] = 0.0
d2v_dy2[:, 0] = d2v_dy2[:, -1] = d2v_dy2 [0, :] = d2v_dy2 [-1, :] = 0.0

# derivative operators
def diff(left, right):
    return (left - right) / dx

def diffx_forward(field):
    c = field[1:-1, 1:-1]
    r = field[1:-1, 2:]
    return diff(c, r)

def diffy_forward(field):
    c = field[1:-1, 1:-1]
    r = field[2:,   1:-1]
    return diff(c, r)

def diffx_backward(field):
    l = field[1:-1, :-2]
    c = field[1:-1, 1:-1]
    return diff(l, c)

def diffy_backward(field):
    l = field[:-2,  1:-1]
    c = field[1:-1, 1:-1]
    return diff(l, c)

# second derivative
def diff2(left, center, right, Delta):
    return (left - 2 * center + right) / Delta**2

def diff2x(field):
    l = field[1:-1, :-2]
    c = field[1:-1, 1:-1]
    r = field[1:-1, 2:]
    return diff2(l, c, r, dx)

def diff2y(field):
    l = field[:-2,  1:-1]
    c = field[1:-1, 1:-1]
    r = field[2:,   1:-1]
    return diff2(l, c, r, dx)

boundary = "dirchlet"

def rhs(h, u, v, is_predictor):
    # compute all spacial derivatives by finte central differences
    Au = (H + h) * u
    Av = (H + h) * v
    if is_predictor:
        dAu_dx [1:-1, 1:-1] = diffx_forward(Au)
        dAv_dy [1:-1, 1:-1] = diffy_forward(Av)
        dh_dx  [1:-1, 1:-1] = diffx_forward(h)
        dh_dy  [1:-1, 1:-1] = diffy_forward(h)
        du_dx  [1:-1, 1:-1] = diffx_forward(u)
        du_dy  [1:-1, 1:-1] = diffy_forward(u)
        dv_dx  [1:-1, 1:-1] = diffx_forward(v)
        dv_dy  [1:-1, 1:-1] = diffy_forward(v)
    else:
        dAu_dx [1:-1, 1:-1] = diffx_backward(Au)
        dAv_dy [1:-1, 1:-1] = diffy_backward(Av)
        dh_dx  [1:-1, 1:-1] = diffx_backward(h)
        dh_dy  [1:-1, 1:-1] = diffy_backward(h)
        du_dx  [1:-1, 1:-1] = diffx_backward(u)
        du_dy  [1:-1, 1:-1] = diffy_backward(u)
        dv_dx  [1:-1, 1:-1] = diffx_backward(v)
        dv_dy  [1:-1, 1:-1] = diffy_backward(v)

    d2u_dx2[1:-1, 1:-1] = diff2x(u)
    d2u_dy2[1:-1, 1:-1] = diff2y(u)
    d2v_dx2[1:-1, 1:-1] = diff2x(v)
    d2v_dy2[1:-1, 1:-1] = diff2y(v)
    laplace_u = d2u_dx2 + d2u_dy2
    laplace_v = d2v_dx2 + d2v_dy2

    # compute rhs
    dh_dt = - dAu_dx - dAv_dy
    du_dt = - u * du_dx - v * du_dy - g * dh_dx - b * u + nu * laplace_u
    dv_dt = - u * dv_dx - v * dv_dy - g * dh_dy - b * v + nu * laplace_v

    if boundary == "neumann":
        return dh_dt, du_dt, dv_dt
    elif boundary == "dirchlet":
        return dh_dt[1:-1, 1:-1], du_dt[1:-1, 1:-1], dv_dt[1:-1, 1:-1]

def do_step(h, u, v):
    print(h.shape)
    # MacCormack method:
    # predictor step
    dh_dt, du_dt, dv_dt = rhs(h, u, v, True)
    if boundary == "neumann":
        h_pred = h + dt * dh_dt
        u_pred = u + dt * du_dt
        v_pred = v + dt * dv_dt
    elif boundary == "dirchlet":
        h_pred = np.zeros(shape)
        u_pred = np.zeros(shape)
        v_pred = np.zeros(shape)
        h_pred[1:-1, 1:-1] = h[1:-1, 1:-1] + dt * dh_dt
        u_pred[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * du_dt
        v_pred[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * dv_dt

    # corrector step
    h_half_step = (h + h_pred) / 2.0
    u_half_step = (u + u_pred) / 2.0
    v_half_step = (v + v_pred) / 2.0
    dh_dt_pred, du_dt_pred, dv_dt_pred = rhs(h_pred, u_pred, v_pred, False)

    if boundary == "neumann":
        h_corr = h_half_step + dt / 2.0 * dh_dt_pred
        u_corr = u_half_step + dt / 2.0 * du_dt_pred
        v_corr = v_half_step + dt / 2.0 * dv_dt_pred
    elif boundary == "dirchlet":
        h_corr = np.zeros(shape)
        u_corr = np.zeros(shape)
        v_corr = np.zeros(shape)
        h_corr[1:-1, 1:-1] = h_half_step[1:-1, 1:-1] + dt / 2.0 * dh_dt_pred
        u_corr[1:-1, 1:-1] = u_half_step[1:-1, 1:-1] + dt / 2.0 * du_dt_pred
        v_corr[1:-1, 1:-1] = v_half_step[1:-1, 1:-1] + dt / 2.0 * dv_dt_pred
    else:
        raise ValueError("invalid bounary condition")

    return h_corr, u_corr, v_corr

time = 0
current_h, current_u, current_v = h0, u0, v0
norm = mpl.colors.Normalize(vmin=np.min(h0), vmax=np.max(h0))
while True:
    plt.clf()
    plt.pcolormesh(x, y, current_h, norm=norm)
    plt.colorbar(norm=norm).set_label("h")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{time:.2f}")
    plt.pause(0.01)
    current_h, current_u, current_v = do_step(current_h, current_u, current_v)
    time += dt

