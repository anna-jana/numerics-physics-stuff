import math, numpy as np, matplotlib.pyplot as plt

np.random.seed(42)
init_lives = 4
ncolors = 3

def step(states, lives):
    new_states = np.empty_like(states)
    new_lives = np.empty_like(lives)
    n, m = states.shape
    for i in range(n):
        for j in range(m):
            self = states[i, j]
            own_lives = lives[i, j]
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    other = states[(i + di) % n, (j + dj) % m]
                    own_lives += np.sign(int(math.remainder(self - other, ncolors)))
                    if own_lives <= 0:
                        self = other
                        own_lives = init_lives
            new_states[i, j] = self
            new_lives[i, j] = own_lives
    return new_states, new_lives

N = 60
states = np.random.randint(0, 3, (N, N))
lives = np.ones((N, N), "int") * init_lives
while True:
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(states)
    plt.subplot(1,2,2)
    plt.imshow(lives)
    states, lives = step(states, lives)
    plt.pause(0.000001)

