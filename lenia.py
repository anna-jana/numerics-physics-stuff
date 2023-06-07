# https://en.wikipedia.org/wiki/Lenia
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

def convolution(kernel_fft, state):
    return ifft2(fft2(state) * kernel_fft).real

def make_step(state, dt, kernel_fft, groth_mapping):
    K = convolution(kernel_fft, state)
    G = groth_mapping(K)
    return np.clip(state + dt * G, 0.0, 1.0)

def kernel_skeleton(kernel_shell, r, beta):
    B = beta.shape[0]
    i = int(B * r)
    if i < len(beta):
        return beta[i] * kernel_shell(np.mod(B * r, 1.0))
    else:
        return 0.0

def make_kernel(kernel_shell, beta, R, size):
    S = 2*size + 1
    xs = np.linspace(-size, size, S)
    K = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            r = np.sqrt(xs[i]**2 + xs[j]**2) / R
            if r <= 1:
                K[i, j] = kernel_skeleton(kernel_shell, r, beta)
    K /= np.sum(K)
    return K

mu = 0.3
sigma = 0.03
groth_mapping = lambda u: 2 * np.maximum(0, 1 - (u - mu)**2 / (sigma**2 * 9) )**4 - 1
groth_mapping = lambda u: np.exp( - (u - mu)**2 / (sigma**2 * 2) ) * 2 - 1
kernel_shell = lambda r: np.exp(4 - 1 / (r * (1 - r)))
kernel_shell = lambda r: (4 * r * (1 - r))**4
beta = np.array([1/2, 2/3, 1.0])
world_radius = 100
dim = world_radius*2 + 1
R = 30
dt = 0.1

np.random.seed(42)
kernel = make_kernel(kernel_shell, beta, R, world_radius)
kernel_fft = fft2(np.roll(kernel, world_radius + 1, (0, 1)))

state = np.random.rand(dim, dim)

S = world_radius*2 + 1
delta_kernel = np.zeros((S, S))
#delta_kernel[S//2, S//2] = 1
delta_kernel[0, 0] = 1
delta_kernel_fft = fft2(delta_kernel)

plt.figure()
step = 0
while True:
    state = make_step(state, dt, kernel_fft, groth_mapping)
    plt.clf()
    plt.pcolormesh(state, vmin=0, vmax=1)
    plt.colorbar(label="state")
    plt.xlabel("x")
    plt.ylabel("y")
    t = dt * step
    step += 1
    plt.title(f"Lenia: {t = :.2f}\n{R = }, {mu = }, {sigma = }, beta = [1/2, 2/3, 1]")
    plt.pause(0.0001)
