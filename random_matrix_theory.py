from pylab import *

num_samples = 1000
for N in [2, 4, 10]:
    samples = np.empty(num_samples)
    for i in range(num_samples):
        M = np.random.randn(N, N)
        M += M.T
        ev = np.linalg.eigvalsh(M)
        n = N//2 - 1
        samples[i] = ev[n + 1] - ev[n]
    plt.hist(samples / np.mean(samples), bins=30, histtype="step", density=True, label=f"N = {N}")
s = np.linspace(np.min(samples), np.max(samples), 400) / np.mean(samples)
plt.plot(s, np.pi/2*s*np.exp(-np.pi/4*s**2), label="analytic result")
plt.title("eigenvalue splitting of random symmetric matrices")
plt.xlabel(r"$\lambda_{n + 1} - \lambda_n$ for $n = N/2$ divide by the mean")
plt.ylabel("probability")
plt.legend()
plt.show()

