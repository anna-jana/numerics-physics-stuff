import sys
import math

import numpy as np
import matplotlib.pyplot as plt

# EOM
# cool: m = 5, N = 10
# do prime numbers for N produce just noise? (no structure?)
m = 5
N = 10

# similar
# m = 1
# N = 2

# a little different
# m = 1
# N = 3

def step(curr, prev):
    nxt = (np.roll(curr, -1) + np.roll(curr, 1) - prev + m*curr) % N # wave/klein gordon
    # nxt = (np.roll(curr, -1) - curr + np.roll(curr, 1)) % N # diff
    # nxt = (np.roll(curr, -1) + np.roll(curr, 1) - prev + m*curr + curr*curr) % N # yukava
    # nxt = (np.roll(curr, -1) + np.roll(curr, 1) - prev - curr + curr*((curr*curr) % N)) % N # wave/klein gordon
    return nxt, curr

# Initial Conditions
SIZE = 1000
particle = np.array([1,2,3,4,5,4,3,2,1]) % N
# particle = np.array([1]) % N
particles = [
    (20, 1),
    #(40, -1),
    (150, -1),
]
prev = np.zeros(SIZE, dtype=np.int)
curr = prev.copy()
for x, v in particles:
    prev[x : x + particle.size] = particle
    curr[x + v : x + v + particle.size] = particle
#prev = np.random.randint(low=0, high=N, size=SIZE)
#curr = np.roll(prev, 1)

# Simulation
STEPS = 1000
hist = [prev]
for i in range(STEPS):
    hist.append(curr)
    curr, prev = step(curr, prev)
hist = np.array(hist)

# Plot
plt.figure(figsize=(20,10))
plt.xlabel("Space", fontsize=15)
plt.ylabel("Time", fontsize=15)
plt.pcolormesh(hist, cmap="spectral")
plt.colorbar()
plt.show()
