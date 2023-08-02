import numpy as np
import matplotlib.pyplot as plt

l = 5
x = np.linspace(-l, l, 500)
a = 1.0
y = a * np.log((a + np.sqrt(a**2 - x**2)) / x) - np.sqrt(a**2 - x**2)
plt.figure()
plt.plot(x, y, "k")
plt.plot(x, -y, "k")
plt.xlabel("x")
plt.ylabel("y")
plt.title("tractrix")
plt.show()

