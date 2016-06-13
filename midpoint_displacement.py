import numpy as np
import matplotlib.pyplot as plt

plt.ion()


def make_nth_size(n):
    return 2 ** n + 1


def midpoint_displacement(size, init_min_val=0.0, init_max_val=10.0, max_error=1.0):
    noise = np.zeros((size, size))
    noise[0, 0] = np.random.uniform(init_min_val, init_max_val)
    noise[0, -1] = np.random.uniform(init_min_val, init_max_val)
    noise[-1, 0] = np.random.uniform(init_min_val, init_max_val)
    noise[-1, -1] = np.random.uniform(init_min_val, init_max_val)
    rec_midpoint_displacement(noise, max_error)
    return noise


def rec_midpoint_displacement(noise, max_error):
    if noise.shape == (2, 2):
        return
    midpoint_index = noise.shape[0] // 2
    noise[0, midpoint_index] = (
        (noise[0, 0] + noise[0, -1]) / 2.0 +
        np.random.uniform(-max_error, max_error)
    )
    noise[midpoint_index, 0] = (
        (noise[0, 0] + noise[-1, 0]) / 2.0 +
        np.random.uniform(-max_error, max_error)
    )
    noise[midpoint_index, -1] = (
        (noise[0, -1] + noise[-1, -1]) / 2.0 +
        np.random.uniform(-max_error, max_error)
    )
    noise[-1, midpoint_index] = (
        (noise[-1, 0] + noise[-1, -1]) / 2.0 +
        np.random.uniform(-max_error, max_error)
    )
    noise[midpoint_index, midpoint_index] = (
        noise[0, midpoint_index] +
        noise[midpoint_index, 0] +
        noise[midpoint_index, -1] +
        noise[-1, midpoint_index]
    ) / 4.0 + np.random.uniform(-max_error, max_error)
    new_max_error = max_error / 2.0
    rec_midpoint_displacement(noise[:midpoint_index + 1, :midpoint_index + 1],
                              new_max_error)
    rec_midpoint_displacement(noise[:midpoint_index + 1, midpoint_index:],
                              new_max_error)
    rec_midpoint_displacement(noise[midpoint_index:, :midpoint_index + 1],
                              new_max_error)
    rec_midpoint_displacement(noise[midpoint_index:, midpoint_index:],
                              new_max_error)


if __name__ == "__main__":
    plt.pcolormesh(midpoint_displacement(make_nth_size(6), max_error=2.0))
    raw_input()
