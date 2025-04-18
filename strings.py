# simulate cosmological topological axion strings in 2D
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from matplotlib import animation
import tqdm

# conversion between different time dependent cosmological quantaties
# H: hubble parameter, t: cosmological time, tau: conformal time, a: scale parameter, log = log(m_r / H),
# m_r is the mass of the radial mode
def log_to_H(l): return 1.0 / np.exp(l)
def H_to_t(H): return 1 / (2*H)
def t_to_H(t): return 1 / (2*t)
def H_to_log(H): return np.log(1/H)
def t_to_tau(t): return -2*np.sqrt(t)
def log_to_tau(log): return t_to_tau(H_to_t(log_to_H(log)))
def t_to_a(t): return np.sqrt(t)
def tau_to_t(tau): return -0.5*(tau)**2
def tau_to_a(tau): return -0.5*tau
def tau_to_log(tau): return H_to_log(t_to_H(tau_to_t(tau)))

# generating random fields
def random_field(N, dx, field_max=1.0/np.sqrt(2), k_max=1.0):
    ks = 2*np.pi * fft.fftfreq(N, dx)
    k1, k2 = np.meshgrid(ks, ks)
    k_mag = np.sqrt(k1**2 + k2**2)
    hat = np.where(k_mag <= k_max,
            np.random.uniform(-field_max, field_max, np.shape(k_mag)) *
            np.exp(1j * np.random.uniform(0.0, 2*np.pi, np.shape(k_mag))),
                0.0)
    field = fft.ifft2(hat)
    return field

# compting the rhs of the eom (comoving coordiates and in conformal time)
def compute_dot_dot(field, tau, dx):
    a = tau_to_a(tau)
    laplace = (sum(np.roll(field, shift=offset, axis=dim) for dim in (0,1) for offset in (-1,1)) - 4 * field) / dx**2
    dot_dot = laplace - field * np.abs(field - 0.5 * a)**2
    return dot_dot

# parameter
np.random.seed(2133434)
nsteps = 100
log_start = 2.0
log_end = 3.0

# uses the required number of grid points to resolve the string core and the hubble scale
L = 1 / log_to_H(log_end)
N = int(np.ceil(L * tau_to_a(log_to_tau(log_end))))
tau_start = log_to_tau(log_start)
tau_end = log_to_tau(log_end)
tau_span = tau_end - tau_start
dtau = tau_span / nsteps
dx = L / N
# setup
field = random_field(N, dx)
field_dot = random_field(N, dx)
tau = tau_start
field_dot_dot = compute_dot_dot(field, tau, dx)
axion_fields = np.empty((nsteps, N, N))

# time stepping using velocity verlet
for i in tqdm.tqdm(range(nsteps)):
    axion_fields[i, :, :] = np.angle(field)
    field += dtau * field_dot + dtau**2/2.0 * field_dot_dot
    field_dot_dot_new = compute_dot_dot(field, tau + dtau, dx)
    field_dot += dtau / 2.0 * (field_dot_dot + field_dot_dot_new)
    field_dot_dot = field_dot_dot_new
    tau += dtau

print(f"{(log_start, log_end) = }")
print(f"{(tau_start, tau_end) = }")
print(f"{L = }")
print(f"{N = }")

# making an animation of the axion over time
plt.figure()
xs = np.linspace(0.0, L, N)
mesh = plt.pcolormesh(xs, xs, axion_fields[0], cmap="twilight", vmin=-np.pi, vmax=np.pi)
title = plt.title(f"tau = {tau_start:.2}")
plt.colorbar(label=r"$\theta$")
plt.xlabel("comoving x [1 / H_end]")
plt.ylabel("comoving y [1 / H_end]")
plt.tight_layout()

def animate(i):
    mesh.set_array(axion_fields[i])
    tau = tau_start + i * dtau
    title.set_text(f"axion field at {tau = :.2}")

my_animation = animation.FuncAnimation(plt.gcf(), animate, interval=1, frames=nsteps)
plt.show()
