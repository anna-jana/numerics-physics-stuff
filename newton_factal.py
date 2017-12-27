import numpy as np
import matplotlib.pyplot as plt

def newton_method(f, df, x0, eps, steps):
    x = x0
    for i in range(steps):
        x_next = x - f(x) / df(x)
        if abs(x_next - x) <= eps:
            return x_next
        x = x_next
    return x

def plot_newton_fractal(f, df, top_left=-2+2.0j, botton_right=2-2.0j, eps=1e-10, steps=100, rows=200, columns=200, cmap="jet"):

    xs = np.linspace(top_left.imag, botton_right.imag, rows)
    ys = np.linspace(top_left.real, botton_right.real, columns)

    fractal = np.array([[newton_method(f, df, x + y*1j, eps, steps) for x in xs] for y in ys])

    plt.subplot(2,2,1)
    plt.title("Real part")
    plt.ylabel("Im")
    plt.pcolormesh(xs, ys, fractal.real, cmap=cmap)
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.title("Imaginary part")
    plt.pcolormesh(xs, ys, fractal.imag, cmap=cmap)
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title("Absolute value")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.pcolormesh(xs, ys, np.abs(fractal), cmap=cmap)
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.xlabel("Re")
    plt.title("Angle")
    plt.pcolormesh(xs, ys, np.angle(fractal), cmap=cmap)
    plt.colorbar()

    plt.tight_layout()


if __name__ == "__main__":
    plot_newton_fractal(lambda z: z**3 - 1, lambda z: 3*z**2)
    plt.show()
