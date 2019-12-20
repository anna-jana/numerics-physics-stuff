# polar method for random normal distributed numbers

import numpy as np
import matplotlib.pyplot as plt

def randn(mean, std):
    while True:
        u = np.random.uniform(-1.0, 1)
        v = np.random.uniform(-1.0, 1)
        s = u**2 + v**2
        if s < 1.0 and s != 0.0:
            break
    s = np.sqrt(-2 * np.log(s) / s)
    return mean + std * u * s

def gaussian(x, mean, std):
    return 1 / np.sqrt(2*np.pi * std**2) * np.exp(- (x - mean)**2 / (2 * std**2))

if __name__ == "__main__":
    m = 0
    s = 1
    samples = [randn(m, s) for _ in range(1000)]
    x = np.linspace(np.min(samples), np.max(samples), 500)
    plt.plot(x, gaussian(x, m, s), label="gaussian")
    plt.hist(samples, histtype="step", density=True, label="samples")
    plt.xlabel("x")
    plt.ylabel("count")
    plt.title("Normal Distributed Random Numbers using the Polar Method")
    plt.legend()
    plt.show()


