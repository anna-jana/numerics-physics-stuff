import numpy as np
import matplotlib.pyplot as plt

def ramer_douglas_peucker(points, eps):
    yield points[0]
    def go(p_s):
        if len(p_s) > 2:
            n = p_s[-1] - p_s[0]
            n = n / np.linalg.norm(n)
            b = p_s[0]
            dist_fn = lambda i: np.linalg.norm((p_s[i] - b) - np.dot(n, p_s[i] - b) * n)
            i = max(range(1, len(p_s) - 1), key=dist_fn)
            if dist_fn(i) >= eps:
                yield from go(p_s[:i+1])
                yield p_s[i]
                yield from go(p_s[i:])
    yield from go(points)
    yield points[-1]

if __name__ == "__main__":
    x_s = np.linspace(0, 2*np.pi, 100)
    y_s = np.cos(6 * x_s) * np.exp(-x_s / 2)
    p_s = list(map(np.array, zip(x_s, y_s)))
    plt.plot(x_s, y_s, "b-")
    x_s_res, y_s_res = zip(*ramer_douglas_peucker(p_s, 0.1))
    plt.plot(x_s_res, y_s_res, "r-")
    plt.show()


