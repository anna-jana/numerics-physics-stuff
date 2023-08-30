import numpy as np
import matplotlib.pyplot as plt

def richardson1(seq):
    n = np.arange(len(seq))
    return (n[:-1] + 1) * seq[1:] - n[:-1] * seq[:-1]

n = np.arange(1, 100)
seq = np.cumsum(1 / n**2)
r1 = richardson1(seq)
r2 = richardson1(r1)
r3 = richardson1(r2)

plt.figure()
plt.plot(n, seq, label="series")
plt.plot(n[:-1], r1, label="$R_1$")
plt.plot(n[:-2], r2, label="$R_2$")
plt.plot(n[:-3], r3, label=r"$R_3$")
plt.axhline(np.pi**2 / 6, color="k", ls="--", label="exact result")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("n")
plt.ylabel("series")
plt.title(r"Richardson extrapolation of $\sum_n 1 / n^2 = \pi^2 / 6$")
plt.legend()
plt.show()
