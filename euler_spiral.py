import numpy as np, matplotlib.pyplot as plt
from scipy.special import fresnel

l = 10
x = np.linspace(-l, l, 500)
S, C = fresnel(x)
plt.plot(S, C)
plt.xlabel("Fresnel S(x)")
plt.ylabel("Fresnel C(x)")
plt.title(f"Euler spiral for x = - {l} .. {l}")
plt.show()
