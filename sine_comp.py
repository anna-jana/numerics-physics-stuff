import numpy as np
import math
import matplotlib.pyplot as plt

def my_sin(x):
    if abs(x) < 0.00017:
        return x
    rec = my_sin(x / -3)
    return 4*rec**3 - 3*rec

def my_sin2(x):
    return sum((-1)**n / math.factorial(2*n + 1) * x**(2*n + 1) for n in range(10))

if __name__ == "__main__":
    x = np.linspace(-2*np.pi, 2.9*np.pi, 300)
    plt.plot(x, np.sin(x), label="numpy")
    plt.plot(x, list(map(my_sin, x)), label="triple angle formula")
    plt.plot(x, list(map(my_sin2, x)), label="power series")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

