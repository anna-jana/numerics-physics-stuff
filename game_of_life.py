import numpy as np

def step(space):
    neighbor_count = sum(np.roll(np.roll(space, dr, 0), dc, 1) for dr in (-1,0,1) for dc in (-1,0,1)) - space
    return ((neighbor_count == 3) | ((space == 1) & (neighbor_count == 2))).astype(np.int)

# space = np.random.choice([0,1], (10,10))

space = np.zeros((5, 5), dtype=np.int)
space[:3, :3] = np.array([[0, 0, 1],
                          [1, 0, 1],
                          [0, 1, 1]])

for i in range(10):
    print(space, "\n")
    space = step(space)
