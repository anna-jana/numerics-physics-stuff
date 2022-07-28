import numpy as np
import scipy.special as sf

def romberg(f, a, b, n_max):
    n = np.arange(n_max)
    h = 1 / 2**n * (b - a)
    R = np.zeros((n_max, n_max)) + np.nan
    R[0, 0] = h[1] * (f(a) + f(b))
    for n in range(1, n_max):
        R[n, 0] = (0.5*R[n - 1, 0] + h[n] *
            sum(f(a + (2*k - 1)*h[n]) for k in range(1, 2**(n - 1) + 1)))
    for n in range(1, n_max):
        for m in range(1, n + 1):
            R[n, m] = R[n, m - 1] + 1 / (4**m - 1) * (R[n, m - 1] - R[n - 1, m - 1])
    return R

print("integral of 2 / np.sqrt(np.pi) * np.exp(- x**2) = erf(1):")
R = romberg(lambda x: 2 / np.sqrt(np.pi) * np.exp(- x**2), 0.0, 1.0, 5)
print("romberg (down: increased h = 1/2^n, to the right: mth richardson extrapoltion):\n", R, sep="")
print("erf(1) =", sf.erf(1))
