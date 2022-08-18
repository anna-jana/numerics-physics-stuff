import numpy as np, matplotlib.pyplot as plt

eps = 1.0
while True:
    new = eps / 2
    if new + 1 == 1: break
    eps = new
h = np.geomspace(eps, 1.0, 300)

def f(x): return np.sin(x) + np.exp(x)
x = 1.0
df_true = np.cos(1) + np.exp(1)
df_one_sided = (f(x + h) - f(x)) / h
df_two_sided = (f(x + h/2) - f(x - h/2)) / h
df_complex   = np.imag(f(x + 1j*h)) / h

plt.figure()
plt.plot(h, np.abs((df_one_sided - df_true) / df_true), label="one sided")
plt.axvline(np.sqrt(eps), label="best one sided", color="tab:blue", ls="--")
plt.plot(h, np.abs((df_two_sided - df_true) / df_true), label="two sided")
plt.axvline(np.cbrt(eps), label="best two sided", color="tab:orange", ls="--")
plt.plot(h, np.abs((df_complex - df_true) / df_true), label="complex continuation")
plt.xlabel("h")
plt.ylabel("relative error")
plt.xscale("log")
plt.yscale("log")
plt.legend(ncol=2, framealpha=1.0)
plt.show()
