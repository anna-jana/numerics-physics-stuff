import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3, 500)
T = 10

A = np.pi*T*x
plt.plot(x, 1 / (4*np.pi*x**2), label="scalar in vacuum")
plt.plot(x, T / (4*np.pi*x) / np.tanh(A), label="scalar at finite T")
plt.plot(x, - 1 / (4*np.pi*x**2) / 2, label="fermion in vacuum")
plt.plot(x, 1 / (4*np.pi*x**2) * (- np.sinh(A) - A * np.cosh(A)) / np.sinh(A)**2, label="fermion at finite T")
plt.ylim(-10, 10)
plt.xlabel("x")
plt.ylabel(r"$\langle \phi(x) \phi(0) \rangle$")
plt.legend()
plt.show()
