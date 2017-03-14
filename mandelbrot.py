import numpy as np
import matplotlib.pyplot as plt

def compute_mandelbrotset(lower=-2.5-2j, upper=1.4+2j, pixel_width=700, max_iterations=80, divergence_radius=3.0):
    pixel_height = int(abs((upper.imag - lower.imag) / (upper.real - lower.real) * pixel_width))
    image = np.zeros((pixel_height, pixel_width), dtype="int")
    divergence_radius_sq = divergence_radius**2
    for row, imag in enumerate(np.linspace(lower.imag, upper.imag, pixel_height)):
        for column, real in enumerate(np.linspace(lower.real, upper.real, pixel_width)):
            z = 0.0 + 0.0j
            c = complex(real, imag)
            for step in xrange(max_iterations):
                z = z*z + c
                if z.real*z.real + z.imag*z.imag > divergence_radius_sq: # abs(z) > divergence_radius
                    break
            image[row, column] = step
    return image

mandel = compute_mandelbrotset()

plt.pcolormesh(mandel, cmap="RdGy")
plt.show()
