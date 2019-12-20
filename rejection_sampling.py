import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def rejection_sampling(pdf, x_min, x_max, pdf_max):
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, pdf_max)
        if y < pdf(x):
            return x

def gaussian(x, mean, std):
    return 1 / np.sqrt(2*np.pi * std**2) * np.exp(- (x - mean)**2 / (2 * std**2))


def randn(mean, std):
    return rejection_sampling(lambda x: gaussian(x, 0.0, 1.0), -10, 10, (2*np.pi)**(-0.5))

if __name__ == "__main__":
    m = 0
    s = 1
    samples = [randn(m, s) for _ in range(1000)]
    x = np.linspace(-10, 10, 500)
    print(simps(gaussian(x, m, s), x))
    to_show = (x >= np.min(samples)) & (x <= np.max(samples))
    plt.plot(x[to_show], gaussian(x[to_show], m, s), label="gaussian")
    plt.hist(samples, histtype="step", density=True, label="samples")
    plt.xlabel("x")
    plt.ylabel("count")
    plt.title("Normal Distributed Random Numbers using Rejection Sampling")
    plt.legend()
    plt.show()
