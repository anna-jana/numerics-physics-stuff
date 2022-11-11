# https://arxiv.org/pdf/1806.04677.pdf
import numpy as np, matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq
from scipy.integrate import solve_ivp

def H_to_tau(H):
    t = 1 / (2*H)
    return t**0.5

def tau_to_H(tau):
    t = tau**2
    return 1 / (2*t)

dim = 2
N = 200
L = 1.0 # this encodes m_r (length is in units of 1 / m_r)
k_max = 0.3
H_start = np.exp(-3.5)
H_end = np.exp(-4.5)

tau_max = N / L

tau_start = H_to_tau(H_start)
tau_end = H_to_tau(H_end)

np.random.seed(42)
xs = np.linspace(0, L, N)
dx = xs[1] - xs[0]
ks_1d = fftfreq(N, dx) * 2*np.pi
ks = np.meshgrid(*([ks_1d]*dim))
fourier_laplacian = - sum(k**2 for k in ks)
k_max_grid = np.sqrt(dim) * np.max(np.abs(ks_1d))
assert k_max < k_max_grid
assert k_max < 1.0 # k is in units of m_r

def random_fourier_field():
    return np.where(fourier_laplacian >= - k_max,
            np.random.uniform(-1/np.sqrt(2), 1/np.sqrt(2), ks[0].shape) + 0j, 0.0)

def rhs(tau, psi_fourier, psi_dot_fourier):
    psi = ifftn(psi_fourier)
    dVdpsi = psi**2 * psi.conj()
    # tau = t**0.5, R = t**0.5, psi_dot_dot_fourier:
    return fourier_laplacian * psi_fourier + (tau / tau_start)**2/2 * psi_fourier - fftn(dVdpsi)

def solver_rhs(tau, y):
    n = len(y) // 2
    psi_fourier = y[:n].reshape([N]*dim)
    psi_dot_fourier = y[n:].reshape([N]*dim)
    return rhs(tau, psi_fourier, psi_dot_fourier)

psi_fourier = random_fourier_field()
psi_dot_fourier = random_fourier_field()
psi_fourier, psi_dot_fourier = psi_fourier.copy(), psi_dot_fourier.copy()

y0 = np.concatenate([psi_fourier.ravel(), psi_dot_fourier.ravel()])

sol = solve_ivp(solver_rhs, (tau_start, tau_end), y0)


# analysis
def plot(psi_fourier, tau):
    psi = ifftn(psi_fourier)
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m_a]")
    ax.set_ylabel("y [m_a]")
    axion = np.angle(psi)
    mesh = ax.pcolormesh(xs, xs, axion, cmap="twilight")
    fig.colorbar(mesh, label=r"$\theta(x, y)$")
    ax.set_title(f"$\\tau = {tau:.2f}$")


# dtau = 1e-3
# nsteps = int(np.ceil((tau_end - tau_start) / dtau))
# max_phi_mag_sq = []
#
# for i in range(nsteps):
#     print(f"step {i + 1} of {nsteps}")
#     # propagate in time
#     tau = tau_start + i * dtau
#     psi_dot_dot_fourier = rhs(tau, psi_fourier, psi_dot_fourier)
#     psi_fourier += dtau * psi_dot_fourier
#     psi_dot_fourier += dtau * psi_dot_dot_fourier
#     # analysis
#     max_phi_mag_sq.append(np.max(np.abs(ifftn(psi_fourier))**2))
#
# taus = tau_start + dtau * np.arange(nsteps)


