import numpy as np
import matplotlib.pyplot as plt

def step(space):
    neighbor_count = sum(np.roll(space, (dr, dc), (0, 1)) for dr in (-1,0,1) for dc in (-1,0,1)) - space
    return ((neighbor_count == 3) | (space & (neighbor_count == 2))).astype("int")

def show(space):
    for i in range(space.shape[0]):
        for j in range(space.shape[1]):
            print("#" if space[i, j] else ".", end="")
        print("")
    print("")

if __name__ == "__main__":
    space = np.zeros((5, 5), dtype="int")
    space[:3, :3] = np.array([[0, 0, 1],
                              [1, 0, 1],
                              [0, 1, 1]])

    for i in range(10):
        show(space)
        space = step(space)

    space = np.random.randint(0,2,(100,100))
    while True:
        plt.clf()
        plt.pcolormesh(space, cmap="Greys")
        space = step(space)
        plt.pause(0.1)

