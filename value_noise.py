import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

def compute_octave(rows, cols, max_val, row_freq, col_freq):
    """ computes an octave of noise for value noise """
    row_step = rows // row_freq
    col_step = cols // col_freq

    noise = np.random.rand(rows, cols)*max_val # HACK

    row_indices = np.arange(rows).repeat(cols).reshape((rows, cols))
    upper_indices = row_indices // row_step * row_step
    lower_indices = np.minimum(upper_indices + row_step, rows - 1)

    col_indices = (
        np.arange(cols).repeat(rows).reshape((cols , rows)).transpose()
    )
    left_indices = col_indices // col_step * col_step
    right_indices = np.minimum(left_indices + col_step, cols - 1)

    upper_left = noise[upper_indices, left_indices]
    upper_right = noise[upper_indices, right_indices]
    lower_left = noise[lower_indices, left_indices]
    lower_right = noise[lower_indices, right_indices]

    from_left = (col_indices - left_indices)/float(col_step)
    upper_val = from_left*(upper_right - upper_left) + upper_left
    lower_val = from_left*(lower_right - lower_left) + lower_left

    from_top = (row_indices - upper_indices)/float(row_step)
    val = from_top*(lower_val - upper_val) + upper_val
    return val

def value_noise(rows, cols, max_val, row_freq, col_freq, num_octaves):
    """
    generates some 2D noise as an numpy array using
    value noise, max_val is divided by two am row_freq and
    col_freq are doubled each octave.
    """
    noise = np.zeros((rows, cols))

    for i in range(num_octaves):
        noise += compute_octave(rows, cols, max_val, row_freq, col_freq)
        max_val /= 2.0
        row_freq *= 2
        col_freq *= 2
    return noise

def normalize_noise(noise):
    """ normalizes an array to values between 0 and 1 """
    return (noise - min(noise))/max(noise)

plt.pcolormesh(value_noise(200, 300, 32.0, 4, 4, 5))
plt.show()
