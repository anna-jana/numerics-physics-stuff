# plot mandelbrot set

import time
import numpy as np
import matplotlib.pyplot as plt

start = time.time()
lower = -2 - 1.3j
upper = 0.6 + 1.3j
width = 400
max_iterations = 80
divergence_radius = 3.0

height = int((upper.imag - lower.imag) / (upper.real - lower.real) * width)
real = np.linspace(lower.real, upper.real, width)
imag = np.linspace(lower.imag, upper.imag, height)
c = real + 1j * imag[:, None]
z = np.zeros(c.shape)
steps = np.zeros(c.shape, dtype=np.int)

for i in range(max_iterations):
    z = z**2 + c
    steps += np.abs(z) < divergence_radius

stop = time.time()
print("took:", stop - start, "seconds")

plt.pcolormesh(real, imag, steps, cmap="RdGy")
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Mandelbrot Set")
plt.show()
