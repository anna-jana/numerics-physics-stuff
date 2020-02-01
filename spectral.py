import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

def diff(x, f, n):
    k = fftfreq(x.size, x[1] - x[0]) * 2 * np.pi
    return np.real(ifft((1j * k)**n * fft(f)))

def solve_heat_eq_pseudo_spectral(x, psi0, Delta_T, D):
    """
    D_t psi = D nabla^2 psi
    """
    k = fftfreq(x.size, x[1] - x[0]) * 2 * np.pi
    return np.real(ifft(np.exp(- D * Delta_T * k**2) * fft(psi0)))
    # D_t psi = D nabla^2 psi + V * psi
    # V = calc_pot(x)
    # return np.real(np.exp(- V / 2 * Delta_T) * ifft(np.exp(- D * Delta_T * k**2) * fft(np.exp(- V / 2 * Delta_T) * psi0)))

if __name__ == "__main__":
    x = np.linspace(0, 2*np.pi, 400)[:-1]
    f = np.sin(x)
    df = diff(x, f, 1)
    ddf = diff(x, f, 2)
    plt.subplot(2, 1, 1)
    plt.plot(x, f, label="f(x) = sin(x)")
    plt.plot(x, df, label="f'")
    plt.plot(x, ddf, label="f''")
    plt.ylim((-1.5, 1.5))
    plt.xlabel("x")
    plt.legend()

    psi0 = np.exp(- (x - np.pi)**2)
    Ts = [0, 1, 2, 3, 4]
    D = 0.5
    plt.subplot(2, 1, 2)
    for T in Ts:
        plt.plot(x, solve_heat_eq_pseudo_spectral(x, psi0, T, D), label="T = %.2f" % T)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel(r"$\psi$")
    plt.title("Diffusion eq. using Spectral Method, D = %.2f" % D)
    plt.show()

