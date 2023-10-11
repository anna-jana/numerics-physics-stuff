import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import zeta

gamma_e = 0.577215664901

def low_temp_J(y):
    return - (y / (2*np.pi))**(3/2) * np.exp(-y)

def high_temp_J(y):
    return (
        - np.pi**2 / 90 + y**2 / 24 - y**3 / (12*np.pi)
        - y**4 / (2*(4*np.pi)**2) * (np.log(y * np.exp(gamma_e) / (4*np.pi)) - 3/4)
        + y**6 * zeta(3) / (3*(4*np.pi)**2)
    )

def full_J(y):
    f = lambda x: x**2 * np.log(1 - np.exp(- np.sqrt(x**2 + y**2)))
    return 1 / (2*np.pi**2) * quad(f, 0.0, np.inf)[0]

plt.figure()
y = np.linspace(0, 10, 400)
plt.plot(y, low_temp_J(y), label=r"low temp. expansion: $-(y/2\pi)^{3/2} e^{-y}$")
plt.plot(y, high_temp_J(y), label=r"high temp. expansion: $-pi^2/90 + y^2/24 +$ ....")
plt.plot(y, [full_J(y_) for y_ in y], label=r"full result: $\frac{1}{2\pi^2} \int_0^\infty dx x^2 \log(1  - e^{- \sqrt{x^2 + y^2}})$")
plt.xlabel("$y = m / T$")
plt.ylabel("$J(y) T^4$")
plt.ylim(-0.12, 0.0)
plt.legend()
plt.title(r"free energy density of a free scalar field $f(m, T) = J(m, T)$")
plt.show()
