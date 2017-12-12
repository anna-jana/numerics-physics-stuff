from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

@jit
def compute_mandelbrotset(lower=-2.5-2j, upper=1.4+2j, pixel_width=700, max_iterations=80, divergence_radius=3.0):
    pixel_height = int(abs((upper.imag - lower.imag) / (upper.real - lower.real) * pixel_width))
    image = np.zeros((pixel_height, pixel_width), dtype=np.int)
    divergence_radius_sq = divergence_radius**2
    for row, imag in enumerate(np.linspace(lower.imag, upper.imag, pixel_height)):
        for column, real in enumerate(np.linspace(lower.real, upper.real, pixel_width)):
            z = 0.0 + 0.0j
            c = complex(real, imag)
            div_step = 0 # for jit
            for step in range(max_iterations):
                div_step = step
                z = z*z + c
                if z.real*z.real + z.imag*z.imag > divergence_radius_sq: # abs(z) > divergence_radius
                    break
            image[row, column] = div_step
    return image


start = time.time()
mandel = compute_mandelbrotset()
stop = time.time()
print("took:", stop - start)

plt.pcolormesh(mandel, cmap="RdGy")
plt.show()
