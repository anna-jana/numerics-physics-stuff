import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp

# i d/dt psi = - nabla^2 / 2m * psi + m phi psi
# nabal^2 phi = 4 pi G |psi|^2 m

# Fourier:
# i d/dt psi_hat = - - k^2 / 2m * psi_hat + m F[phi * psi]
# - k^2 phi_hat = 4 pi G m F[|psi|^2]

# free solution:
# i d/dt psi_hat = H psi_hat = k^2 / 2m psi_hat
# psi_final = e^(-i dt H) psi0 hat = exp(-i dt k^2 / 2m) psi0 hat

L = 1.0
N = 100
m = 1.0
G = 1.0
sigma0 = 1e-2
A0 = 1.0
tspan = 1e-2

x = np.linspace(0, L, N+1)[:-1]
dx = x[1] - x[0]
k = fftfreq(N, dx) * 2*np.pi
psi0 = A0 * np.exp(- (x - L/2)**2 / sigma0)

def rhs(t, psi_hat):
    psi = ifft(psi_hat)
    phi_hat = - (4*np.pi*G*m) / k**2 * fft(np.abs(psi)**2)
    phi_hat[0] = 0.0 # k = 0 (zero) mode is arbitary (but undefined after dividing by k^2)
    phi = ifft(phi_hat)
    d_psi_hat_dt = -1j * (k**2 / (2*m) * psi_hat + m * fft(phi * psi))
    return d_psi_hat_dt

sol = solve_ivp(rhs, (0, tspan), fft(psi0), method="BDF", rtol=1e-5, atol=1e-5)
psi_final = ifft(sol.y[:, -1])

psi_final_free = ifft(np.exp(-1j * tspan * k**2 / (2*m) * fft(psi0)))

plt.figure()
plt.plot(x, psi0, label="initial")
plt.plot(x, psi_final, label=f"final {tspan = }")
plt.plot(x, psi_final_free, label="final free solution")
plt.xlabel("x")
plt.ylabel("psi(x)")
plt.title("Schroedinger Poisson Equation")
plt.legend()
plt.show()
