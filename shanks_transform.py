import matplotlib.pyplot as plt
import numpy as np

def shanks_transform(seq):
    A_n_minus_1, A_n, A_n_plus_1 = seq[:-2], seq[1:-1], seq[2:]
    return A_n_plus_1 - (A_n_plus_1 - A_n)**2 / ((A_n_plus_1 - A_n) - (A_n - A_n_minus_1))

n = np.arange(100)
pi_series = np.cumsum(4 * (-1)**n / (2*n + 1))
S = shanks_transform(pi_series)
S2 = shanks_transform(S)
S3 = shanks_transform(S2)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(n, pi_series, label="$A_n$")
plt.plot(n[1:-1], S, label="$S_n$")
plt.xlabel("n")
plt.ylabel("series")
plt.legend()
plt.title(r"Shanks transform of $\sum_k 4 (-1)^k / (2k + 1) = \pi$")
plt.subplot(2, 1, 2)
plt.plot(n, np.abs(np.pi - pi_series) / np.pi, label="$A_n$")
plt.plot(n[1:-1], np.abs(np.pi - S) / np.pi,   label="$S(A)_n$")
plt.plot(n[2:-2], np.abs(np.pi - S2) / np.pi,  label="$S^2(A)_n$")
plt.plot(n[3:-3], np.abs(np.pi - S3) / np.pi,  label="$S^3(A)_n$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.xlabel("n")
plt.ylabel("relative error")
plt.show()
