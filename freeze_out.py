import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# computes relic density of freeze out dark matter

# parameter: mass of the wimp m
#            thermally averaged crosssection < sigma v > for scattering with SM particle (we call this just sigma in the code)

# everything is in units of MeV

# Y = n / s
# x = m / T

# https://arxiv.org/pdf/1606.07494.pdf
log10_T_in_MeV, g_rho, g_rho_over_g_s = np.array(
         ((0.00, 10.71, 1.00228),
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

g_rho_interp = interp1d(log10_T_in_MeV, g_rho, kind="cubic", bounds_error=False, fill_value=(g_rho[0], g_rho[-1]))
g_rho_over_g_s_interp = interp1d(log10_T_in_MeV, g_rho_over_g_s, kind="cubic", bounds_error=False, fill_value=(g_rho_over_g_s[0], g_rho_over_g_s[-1]))
g_s_interp = lambda log10_T_in_MeV: g_rho_interp(log10_T_in_MeV) / g_rho_over_g_s_interp(log10_T_in_MeV)

g = 2 # dofs of the wimp
M_pl = 1e19 / 8*np.pi * 1e3 # [MeV]

def calc_entropy_density(T):
    # s = 2pi**2/45 * g_(*,s)(T) * T**3
    return  2*np.pi**2/45 * g_s_interp(np.log10(T)) * T**3

def calc_H(x):
    # 3*M_pl^2*H^2 = pi^2/30 g_*(T) T^4
    T = m / x
    return np.pi/np.sqrt(45)/M_pl * np.sqrt(g_rho_interp(np.log10(T))) * T**2

def calc_Y_eq(x, m):
    T = m / x
    n_eq = g * (m*T/2*np.pi)**(3/2) * np.exp(-x) # for m >> T and mu = 0
    s = calc_entropy_density(T)
    return n_eq / s

# d s / dt = - 3 * x * H bc s * a**3 = const
def boltzmann_rhs(log_x, log_Y, m, sigma):
    x = np.exp(log_x)
    Y = np.exp(log_Y)
    T = m / x
    s = calc_entropy_density(T)
    H = calc_H(x)
    Y_eq = calc_Y_eq(x, m)
    d_log_Y_d_log_x = - s * sigma / H * (Y**2 - Y_eq**2) / Y
    return d_log_Y_d_log_x

def solve_boltzmann(m, sigma, inital_Y, initial_x, final_x):
    # d Y / d x = - x * s < sigma v > / H(m) * (Y**2 - Y_eq*2)
    sol = solve_ivp(boltzmann_rhs, (np.log(initial_x), np.log(final_x)), (np.log(inital_Y),), args=(m, sigma), dense_output=True, method="BDF")
    assert sol.success
    return lambda x: np.exp(sol.sol(np.log(x)))[0, :]

alpha = 0.01
m = 100 * 1e3 # [MeV]
sigma = alpha**2 / m**2
xs = np.geomspace(1, 100, 500)
Y_eqs = calc_Y_eq(xs, m)
Y0 = Y_eqs[0]
sol = solve_boltzmann(m, sigma, Y0, xs[0], xs[-1])
Y = sol(xs)

plt.figure()
plt.plot(xs, Y / Y0, label="freeze out")
plt.plot(xs, Y_eqs / Y0, label="equilirium")
plt.xlabel("x = m/T")
plt.ylabel("Y / Y_0")
plt.title(f"m = {m:.2e} MeV, <sigma v> = {sigma:.2e} MeV^-2")
plt.xscale("log")
plt.yscale("log")
plt.ylim(min(Y) * 1e-2, 2)
plt.legend()
plt.show()
