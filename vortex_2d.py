import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import rfftn, irfftn, rfftfreq, fftfreq
import tqdm

# pseudo spectral navier stokes solver
def rhs(u_hat, v_hat, Re, k_x, k_y):
    u = irfftn(u_hat)
    v = irfftn(v_hat)
    advection_x = rfftn(u * irfftn(1j * k_x * u_hat) + v * irfftn(1j * k_y * u_hat))
    advection_y = rfftn(u * irfftn(1j * k_x * v_hat) + v * irfftn(1j * k_y * v_hat))

    friction_x = (- k_x**2 - k_y**2) * u_hat / Re
    friction_y = (- k_x**2 - k_y**2) * v_hat / Re

    d_u_hat_dt = - advection_x + friction_x
    d_v_hat_dt = - advection_y + friction_y

    return d_u_hat_dt, d_v_hat_dt

def pressure_projection_inplace(u_hat, v_hat, dt, k_x, k_y):
    div_v = 1j * (k_x * u_hat + k_y * v_hat)
    p_hat = - div_v / (-k_x**2 - k_y**2) / dt
    p_hat[0, 0] = 0.0
    u_hat += 1j * k_x * dt * p_hat
    v_hat += 1j * k_y * dt * p_hat

def compute_vorticity(u_hat, v_hat, k_x, k_y):
    dudy = irfftn(- 1j * k_y * u_hat)
    dvdx = irfftn(- 1j * k_x * v_hat)
    vorticity = dudy - dvdx
    my_min = np.min(vorticity)
    return 2 * (vorticity - my_min) / (np.max(vorticity) - my_min) - 1

def random_field(k_x, k_y, k_max, V):
    return N * np.where(k_x**2 + k_y**2 < k_max**2,
        np.random.uniform(0.0, V, size=np.shape(k_x)) *
        np.exp(-1j * np.random.uniform(0.0, 2*np.pi, size=np.shape(k_x))), 0.0)

# simulation parameters
N = 50
dt = 1e-4
tmax = 1.0
Re = 1000.0
save_dt = 1e-2
L = 1.0
V = 10.0
k_max_percent = 1 / 4.0

# setup
save_every_nsteps = int(save_dt / dt)
data = []
delta = L / N # periodic bc
nsteps = int(np.ceil(tmax / dt))

# fields
k_max_grid = 2*np.pi / L * N / 2.0
k_max = k_max_grid * k_max_percent
k_x_flat = rfftfreq(N, delta) * 2*np.pi
k_y_flat = fftfreq(N, delta) * 2*np.pi
k_x, k_y = np.meshgrid(k_x_flat, k_y_flat)
np.random.seed(2123)
u_hat = random_field(k_x, k_y, k_max, V)
v_hat = random_field(k_x, k_y, k_max, V)
d_u_hat_dt_prev = None
d_v_hat_dt_prev = None

# time integration loop
for i in tqdm.tqdm(range(nsteps)):
    d_u_hat_dt, d_v_hat_dt = rhs(u_hat, v_hat, Re, k_x, k_y)

    # time step
    if i == 0:
        # forward euler
        u_hat += dt * d_u_hat_dt
        v_hat += dt * d_v_hat_dt
    else:
        # adams bashforth 2
        u_hat += dt * (3 * d_u_hat_dt - d_u_hat_dt_prev) / 2.0
        v_hat += dt * (3 * d_v_hat_dt - d_v_hat_dt_prev) / 2.0
    d_u_hat_dt_prev = d_u_hat_dt
    d_v_hat_dt_prev = d_v_hat_dt

    pressure_projection_inplace(u_hat, v_hat, dt, k_x, k_y)

    assert np.isfinite(u_hat[1, 1])

    if i % save_every_nsteps == 0:
        data.append((
            i * dt,
            compute_vorticity(u_hat, v_hat, k_x, k_y),
            irfftn(u_hat), irfftn(v_hat),
        ))

# create animation of the flow
fig = plt.figure()
plt.xlabel("x")
plt.ylabel("y")
x = np.linspace(0, L, N)
t, zeta, u, v = data[0]
mesh = plt.pcolormesh(x, x, zeta, vmin=-1, vmax=+1)
plt.colorbar(label="vorticity (normalized)")
quiver = plt.quiver(x, x, u, v)
title = plt.title("")

def animate(i):
    t, zeta, u, v = data[i]
    mesh.set_array(zeta)
    quiver.set_UVC(u, v)
    title.set_text(f"2D Flow\nt = {t:.3}")

animation = FuncAnimation(fig, animate, interval=100, frames=len(data))
plt.show()
