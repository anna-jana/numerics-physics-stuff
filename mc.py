import numpy as np
import matplotlib.pyplot as plt

print("integrate pi")
# direct sampling
N = 1000
x = np.random.rand(N)
y = np.sqrt(1 - x**2)
I = 4 * 1 * np.mean(y)
I_err = 4 * 1 * np.std(y) / np.sqrt(N)
print(I, "+/-", I_err)

x = np.random.rand(N)
y = np.random.rand(N)
r = x**2 + y**2
inside = r < 1
pi_approx = 4 * np.mean(inside)
print("I =", pi_approx, "+/-", 1/np.sqrt(np.sum(inside)))
print("N_acc =", np.sum(inside))


plt.figure()
plt.plot(x[inside], y[inside], "gx", label="inside")
plt.plot(x[~inside], y[~inside], "rx", label="outside")
xs = np.linspace(0, 1, 300)
plt.plot(xs, np.sqrt(1 - xs**2), "k-", label="circle")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title(r"$\pi \approx$ " + str(pi_approx))

print("importance sampling")
import sympy as sp
from sympy.abc import x, y
sp.init_printing()
x_prime = sp.symbols("x_prime")

x_min, x_max = 0, 10
N = 100000
f = x*sp.exp(-x)
g = 2/sp.E*sp.exp(-x/2)
G = g.integrate((x, x_min, x_prime)).subs(x_prime, x)
G_inv = sp.solve(sp.Eq(G, y), x)[0].subs(y, x)

print("f:", f)
print("g:", g)
print("G:", G)
print("G^-1:", G_inv)

f = sp.lambdify(x, f, "numpy")
g = sp.lambdify(x, g, "numpy")
G = sp.lambdify(x, G, "numpy")
G_inv = sp.lambdify(x, G_inv, "numpy")

A = (G(x_max) - G(x_min))
x_g_samples = G_inv(G(x_min) + np.random.rand(N)*A)
y_samples = np.random.uniform(0, g(x_g_samples))
N_acc = np.sum(y_samples < f(x_g_samples))

print("acceptence ratio:", N_acc / N)
print("integral:", N_acc / N * A, 1/np.sqrt(N))

from scipy.integrate import quad
print("check with quad from scipy:", quad(f, x_min, x_max))

plt.figure()
x = np.linspace(x_min, x_max, 400)
plt.plot(x, f(x), label="f")
plt.plot(x, g(x), label="g")
plt.hist(x_g_samples, density=True, label="samples", bins=30, histtype="step")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
