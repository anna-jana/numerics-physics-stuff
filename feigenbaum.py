from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import time

def logistic_map_fixpoints(x0, k, steps, to_take):
    xs = np.zeros(steps)
    xs[0] = x0
    for i in range(1, steps):
        xs[i] = xs[i - 1]*k*(1.0 - xs[i - 1])
    fixpoints = list(set(xs[-to_take:]))
    return fixpoints

def bifurcation_logistic_map(k_start, k_stop, k_step, steps, to_take, x0):
    ks = []
    fix_xs = []
    for k in np.arange(k_start, k_stop, k_step):
        fixpoints = logistic_map_fixpoints(x0, k, steps, to_take)
        ks.extend([k] * len(fixpoints))
        fix_xs.extend(fixpoints)
    return ks, fix_xs

def plot_feigenbaum(k_start=0.0, k_stop=4.0, k_step=0.001, steps=100, to_take=10, x0=0.1, markersize=1.0):
    ks, xs = bifurcation_logistic_map(k_start, k_stop, k_step, steps, to_take, x0)
    plt.xlim(k_start, k_stop)
    plt.plot(ks, xs, ".b", markersize=markersize)
    plt.title(r"Bifurctation diagram for the logistic map $x_{n + 1} = x_n k (1 - x_n)$")
    plt.xlabel("k")
    plt.ylabel("x")

if __name__ == "__main__":
    start = time.time()
    plot_feigenbaum(k_start=2.8, k_step=0.0001)
    stop = time.time()
    compute_time = stop - start
    print("compute time:", compute_time, "seconds")
    plt.show()
