import itertools
import numpy as np
import matplotlib.pyplot as plt


# E(x) = - log(P(x))
def metropolis(E, T, r, x):
    while True:
        q = np.random.uniform(-1, 1, x.shape)
        y = x + r * q
        delta_E = E(y) - E(x)
        # in general
        p = min([1, np.exp(- delta_E / T)])
        if np.random.rand() < p:
            x = y
        yield x

if __name__ == "__main__":
    steps = itertools.islice(metropolis(lambda x: x @ x / 2, 100.0, 10.0, np.array([1.0])), 1000)
    plt.hist([s[0] for s in steps], 20, histtype="step")
    plt.show()
