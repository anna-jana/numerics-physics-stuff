import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.signal import windows

def run(rule, w, nsteps):
    hist = np.empty((nsteps + 1, w.shape[0]), dtype=bool)
    hist[0, :] = w
    for i in range(nsteps):
        l = np.roll(hist[i, :], 1)
        r = np.roll(hist[i, :], -1)
        patter_index =  4*l + 2*hist[i] + r
        hist[i + 1, :] = (rule & (1 << patter_index)).astype("bool")
    return hist

# N = 2^winsize
# p = 1/N
# sum(- p_i * log2(p_u)) = N * - 1 / N log2(1 / N) = log2(N) = log2(2^winsize) = winsize
def compute_entropy(w, winsize=3):
    pattern_indicies = sum(2**i * w[i:w.shape[0]- winsize + i] for i in range(winsize))
    counts = np.array([np.sum(pattern_indicies == i) for i in range(2**winsize)])
    props = counts / np.sum(counts)
    return np.sum(np.where(props != 0.0, - np.log2(props) * props, 0.0)) / winsize

def gen_samples(rule, size, nsteps, init):
    if init == "onehot":
        w = np.zeros(size, dtype=bool)
        w[0] = True
    elif init == "random":
        w = np.random.randint(0, 2, size)
    else:
        raise ValueError("invalid init form")

    history = run(rule, w, nsteps)
    entropy = np.array([compute_entropy(l) for l in history])

    return history, entropy

def plot_evolution(rule, size, nsteps, init):
    history, entropy = gen_samples(rule, size, nsteps, init)
    spectrum = fft.rfft(windows.blackman(len(entropy), False) * entropy)

    fig = plt.figure(layout="constrained")
    fig.suptitle(f"Rule: {rule}")
    axs = fig.subplots(3, 1) # , sharex=True)

    axs[0].pcolormesh(history.T, cmap="gray")
    axs[0].set_ylabel("cell index")
    axs[0].set_xlabel("step")

    axs[1].plot(entropy)
    axs[1].set_ylabel("entropy")
    axs[1].set_xlabel("step")

    axs[2].plot(spectrum.imag)
    axs[2].set_ylabel("entropy spectrum")
    axs[2].set_xlabel("frequency")

def bits(byte):
    return list(reversed([(byte >> i) & 1 for i in range(8)]))

def swap_bits(byte, i, j):
    # get bits
    bit_at_i = (byte >> i) & 1
    bit_at_j = (byte >> j) & 1
    # clear bits
    byte &= ~(1 << i)
    byte &= ~(1 << j)
    # set bits
    byte |= bit_at_i << j
    byte |= bit_at_j << i
    return byte

# mirror the neighborhood around the center (vertical) axis
def mirror(byte):
    byte = swap_bits(byte, 1, 4)
    byte = swap_bits(byte, 6, 3)
    return byte

def reverse_bits(byte):
    out = 0
    for i in range(8):
        out <<= 1
        out |= byte & 1
        byte >>= 1
    return out

# exchange 1 and 0
def exchange(byte):
    return reverse_bits(~byte)

def compute_unique_rules():
    unique_rules = []
    for rule in range(2**(2**3)):
        rule = np.uint8(rule)
        if all(rule != mirror(unique_rule) and
               rule != exchange(unique_rule) and
               rule != exchange(mirror(unique_rule))
                for unique_rule in unique_rules):
            unique_rules.append(rule)
    return unique_rules

def show_all():
    for rule in compute_unique_rules():
        plot_evolution(rule, 100, "random")
        plt.show()
