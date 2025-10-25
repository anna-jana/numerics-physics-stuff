################################################################################################
################################# cosmological pertubation theory ##############################
################################################################################################
# ## zeroth order quantaties:
# * cold dark matter density rho_cdm
# * baryon densty rho_b
# * photon temperature T_gamma (in 1MEV)
# * neutrino temperature T_nu
# ## first order quantaties (delta Q = Q(position) - Q_mean, contrast means delta Q / Q_mean, for any quantiy Q):
# * dark matter density contrast delta_CDM
# * dark matter streaming velocity u_CDM
# * baryon density contrast delta_B
# * baryon streaming velocity u_B
# * photon temperature contrast Theta
# * neutrino temperature contrast N
# ## gravity: ds^2 = - (1 + 2*Psi) dt^2 + a(t)^2 (1 + 2*Phi) dx_i^2
# * conformal time eta d_eta = dt / a(t)
# * scale factor a, convention a(today) = 1
# * scalar potentials Psi, Phi
# ## units:
# * per definition:
#   * [delta] = [Theta] = [N] = [u] = 1
#   * [mu] = 1
# * choice:
#   * [Psi] = [Phi] = 1
#   * [a] = 1
# * units for space-time:
#   * eta -> eta / H0
#   * k -> H0 k
#   * drops out for the matter equations as they are first order and linear
# * units for energy:
#   * rho_i -> T_CMB**4 * rho_i
#   * T_i -> T_CMB T_i
# * in the gravity equsations unit are combined into dimensionless parameters

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from astropy import constants as c
from astropy import cosmology

####################################### momentum distribution #######################################
def compute_monopol(f):
    return np.mean(f)

num_mu_samples = 30
mu = np.linspace(-1, +1, num_mu_samples)

def compute_dipol(f):
    return 1j * np.mean(mu * f)

############################################ constants ##############################################
# in physical units
T_CMB = (cosmology.Planck15.Tcmb0 * c.k_B).to("MeV").value
G = (c.G / c.hbar / c.c**5).to("1/MeV^2").value
M_pl =  2.435e18 * 1e3 # [MeV] # https://en.m.wikiversity.org/wiki/Physics/Essays/Fedosin/Planck_mass
H0 = (cosmology.Planck15.H0 * c.hbar).to("MeV").value
rho_critical_today = (cosmology.Planck15.critical_density0 * c.c**2 * (c.hbar * c.c)**3).to("MeV**4").value

# choice of code units
energy_unit = T_CMB # in MeV
spacetime_unit = H0 # in MeV

rho_cdm_0 = cosmology.Planck15.Odm(0.0) * rho_critical_today / energy_unit**4
rho_b_0 = cosmology.Planck15.Ob(0.0) * rho_critical_today / energy_unit**4

sigma_T = (c.sigma_T/c.c**2/c.hbar**2).to("1/MeV^2").value / energy_unit**2

param1 = energy_unit**2 / (M_pl / energy_unit) / spacetime_unit
param2 = G / spacetime_unit**2 * energy_unit**4

########################################### entropy ################################################
g_rho_today = 2 + 7/8 * 6 * (4 / 11)**(4/3)
g_s_today = 2 + 7/8 * 6 * 4 / 11

T_ep_annihilation_end = 0.005 # [MeV] NOTE: starts at 0.5 MeV

log10_T_in_MeV, g_rho, g_rho_over_g_s = np.array(
        # NOTE: added to account for e+ e- annihlation
        ((np.log10(T_ep_annihilation_end), g_rho_today, g_rho_today / g_s_today),
          (0.00, 10.71, 1.00228),
          (0.50, 10.74, 1.00029),
          (1.00, 10.76, 1.00048),
          (1.25, 11.09, 1.00505),
          (1.60, 13.68, 1.02159),
          (2.00, 17.61, 1.02324),
          (2.15, 24.07, 1.05423),
          (2.20, 29.84, 1.07578),
          (2.40, 47.83, 1.06118),
          (2.50, 53.04, 1.04690),
          (3.00, 73.48, 1.01778),
          (4.00, 83.10, 1.00123),
          (4.30, 85.56, 1.00389),
          (4.60, 91.97, 1.00887),
          (5.00, 102.17, 1.00750),
          (5.45, 104.98, 1.00023),)).T

g_rho_interp = interp1d(
        log10_T_in_MeV, g_rho,
        kind="cubic", bounds_error=False, fill_value=(g_rho[0], g_rho[-1]))
g_rho_over_g_s_interp = interp1d(
        log10_T_in_MeV, g_rho_over_g_s,
        kind="cubic", bounds_error=False, fill_value=(g_rho_over_g_s[0], g_rho_over_g_s[-1]))

def compute_entropy_dofs(T):
    log10_T_in_MeV = np.log10(T)
    return g_rho_interp(log10_T_in_MeV) / g_rho_over_g_s_interp(log10_T_in_MeV)

cbrt_g_star_entropy_today = compute_entropy_dofs(T_CMB)

########################################## temperatures ##########################################
# in energy_unit units
def compute_photon_temperature(a):
    def goal(T):
        return T_CMB * cbrt_g_star_entropy_today / (a * T * np.cbrt( compute_entropy_dofs(energy_unit * T) )) - 1.0
    sol = root_scalar(goal, method="newton", x0=T_CMB / a)
    assert sol.converged
    return sol.root

# in energy_unit units
def compute_neutrino_temperature(T_gamma):
    if T_gamma > T_ep_annihilation_end / energy_unit:
        return T_gamma
    else:
        return (4 / 11)**(1/3) * T_gamma

######################################## equations of motion ########################################
nvars = 6

def compute_pertubation_rhs(eta, y, k):
    a, Phi, delta_CDM, u_CDM, delta_B, u_B = y[:nvars]
    Theta = y[nvars:nvars + num_mu_samples]
    N = y[nvars + num_mu_samples:]
    a = np.real(a)

    ###### background #######
    # energy densities
    rho_cdm = rho_cdm_0 / a**3
    rho_b = rho_b_0 / a**3

    T_gamma = compute_photon_temperature(a)
    rho_gamma = np.pi**2 / 15 * T_gamma**4

    T_nu = compute_neutrino_temperature(T_gamma)
    rho_nu = 3 * 7/8 * np.pi**2 / 15 * T_nu**4 # NOTE: assume massless neutrinos (okay for early times)

    rho_total = rho_cdm + rho_b + rho_gamma + rho_nu # NOTE: we ignore dark energy

    # friedmann equations:
    d_a_d_eta = a**2 * np.sqrt(rho_total / 3) * param1

    ######## scalar pertubations ########
    # multipol moments for radtion:
    Theta_0 = compute_monopol(Theta)
    Theta_1 = compute_dipol(Theta)
    N_0 = compute_monopol(N)
    N_1 = compute_dipol(N)

    # gravity:
    Psi = - 32 * param2 * a**2 * (rho_gamma * Theta_1 + rho_nu * N_1) / k**2 - Phi

    right = 4*np.pi * param2 * a**2 * (rho_cdm * delta_CDM + rho_b * delta_B +
                               4 * (rho_gamma * Theta_0 + rho_nu * N_0))
    d_Phi_d_eta = (right - k**2 * Phi) / (3 * d_a_d_eta / a) + d_a_d_eta / a * Psi

    # matter:
    n_e = 4 / np.pi**2 * T_gamma**3 # TODO: correct? ever non-relativistic?
    d_tau_d_eta = - n_e * sigma_T * a

    # cold dark matter:
    d_delta_CDM_d_eta = - 1j*k*u_CDM - 3*d_Phi_d_eta
    d_u_CDM_d_eta = - d_a_d_eta / a * u_CDM - 1j*k*Psi

    # baryons:
    d_delta_B_d_eta = - 1j*k*u_B - 3*d_Phi_d_eta
    d_u_B_d_eta = - d_a_d_eta / a * u_B - 1j*k*Psi + d_tau_d_eta * 4 * rho_gamma / (3 * rho_b) * (3j * Theta_1 + u_B)

    # radiation:
    d_Theta_d_eta = - 1j*k*mu * Theta - d_Phi_d_eta - 1j*k*mu * Psi - d_tau_d_eta * (Theta_0 - Theta + mu * u_B)
    d_N_d_eta = - 1j*k*mu*N - d_Phi_d_eta - 1j*k*mu * Psi

    return np.concatenate([
        [d_a_d_eta, d_Phi_d_eta, d_delta_CDM_d_eta, d_u_CDM_d_eta, d_delta_B_d_eta, d_u_B_d_eta],
        d_Theta_d_eta, d_N_d_eta])

###################################### initial conditions #######################################
A_s = 1.0 # amplitude of initial scalar pertubations
k_0 = 0.05 # [1/Mpc]
n_s = 1.0 # spectral index of scalar pertubations
# ic's
def compute_initial_distribution(k):
    # TODO
    return A_s * (k / k_0)**(n_s - 1.0)

def compute_anisotropies():
    # TODO
    pass

############################################ solve odes ##################################################
nks = 100
k_max = 0.9
ks = np.linspace(0, k_max, nks)
# TODO
eta_inf = 1.0 # end of inflatipn
eta_star = 2.0 # surface of last scattering
a_inf = 1e-7
y0 = np.ones(nvars + 2*num_mu_samples, dtype=np.complex128)
y0[0] = a_inf

def solve(k):
    sol = solve_ivp(compute_pertubation_rhs, (eta_inf, eta_star), y0, args=(k,), method="BDF")
    assert sol.success
    return sol

def plot(data):
    plt.figure()
    plt.show()

