import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vegas

# the path integral quantum monte carlo method:
def calc_euclidean_action(config, x_i, x_f, m, a, V_fn):
    kin = m/(2*a**2) * (np.sum((config[1:] - config[:-1])**2)
                   + (x_i - config[0])**2 + (config[-1] - x_f)**2) # boundary
    pot = np.sum(V_fn(config)) + V_fn(x_i) + V_fn(x_f)
    return a * (kin + pot)

def make_path_integrant(m, a, N, x_i, x_f, V_fn):
    A = (m / (2*np.pi*a))**(N / 2)
    return lambda config: (
        A * np.exp(- calc_euclidean_action(config, x_i, x_f, m, a, V_fn))
    )

# use the vegas library to integrate the integrals using an adaptive monte carlo integrator
def calc_path_integral(m, x_i, x_f, T, N, V_fn):
    a = T / N
    span = 3
    vint = vegas.Integrator([(-span, span)] * (N - 1))
    I_fn = make_path_integrant(m, a, N, x_i, x_f, V_fn)
    vint(I_fn, nitn=100, neval=1000)
    res = vint(I_fn, nitn=10, neval=10000)
    #assert res.var / res.val < 1e-3
    return res.mean, res.sdev

# example (harmonic oscillator):
# parameters
m = 1.0
N = 9
T = 4.0
x_min = 0
x_max = 2
x_vals = np.linspace(x_min, x_max, 11)
x_range = np.linspace(x_min, x_max, 100)

# analytical solution
E0 = 1/2
wave_fn = lambda x: np.exp(- x**2 / 2) / np.pi**(1/4)
calc_prop_analytic = lambda x: np.abs(wave_fn(x))**2 * np.exp(- E0 * T)

# numerical solution for this example
def calc_prop(x, V_fn):
    return calc_path_integral(m, x, x, T, N, V_fn)
def V_harm(x):
    return x**2 / 2
ans = np.array([calc_prop(x, V_harm) for x in tqdm(x_vals)])

# plot the result
prop_vals, prop_err = ans.T
analytic = calc_prop_analytic(x_range)
plt.plot(x_range, analytic, label="analytic")
norm = 1 # 1 / prop_vals[0] * analytic[0] # cheat by normalizing by the first analytical result
plt.errorbar(x_vals, prop_vals * norm, fmt="o", yerr=prop_err, label="numerical path integral")
plt.xlim(0,2)
plt.ylim(0,0.1)
plt.xlabel("x")
plt.ylabel(r"$\langle x \vert e^{- \tilde{H} T} \vert x \rangle$")
plt.legend()
plt.show()
