import matplotlib.pyplot as plt, numpy as np, itertools

np.random.seed(42); plt.ion()

N = 25
n = 4
sand = np.zeros((N, N))

while True:
    fill_x, fill_y = np.random.randint(0, N, 2)
    sand[fill_x, fill_y] += 1

    can_be_too_full = [(fill_x, fill_y)]
    while can_be_too_full:
        kth = np.random.randint(0, len(can_be_too_full))
        x, y = coord = can_be_too_full[kth]
        del can_be_too_full[kth]
        if sand[coord] >= n:
            sand[coord] -= n
            for dx, dy in itertools.product((-1,0,1), repeat=2):
                if (dx == 0) != (dy == 0):
                    neighbor_coord = (x + dx) % N, (y + dy) % N
                    sand[neighbor_coord] += 1
                    can_be_too_full.append(neighbor_coord)

    plt.clf()
    plt.pcolormesh(sand, vmin=0, vmax=n-1)
    plt.colorbar()
    plt.pause(0.01)



