import numpy as np
import matplotlib.pyplot as plt

# k = np.linspace(min_k, max_k, num_k)
k = np.arange(2.8, 4.0, 0.0001)
x = np.empty((10000, k.size))
x[0, :] = 0.01
for i in range(x.shape[0] - 1):
    x[i + 1, :] = k * x[i, :] * (1 - x[i, :])
keep = 100
x = x[-keep:, :].T.reshape(-1)
k = k.repeat(keep)
plt.plot(k, x, ".b", markersize=0.1)
plt.xlabel("k")
plt.ylabel("x fixpoints")
plt.title(r"Bifurctation diagram for the logistic map $x_{n + 1} = x_n k (1 - x_n)$")
plt.show()
