import matplotlib.pyplot as plt
import numpy as np

LEFT = (0, -1)
RIGHT = (0, 1)
UP = (-1, 0)
DOWN = (1, 0)

WHITE = 0
BLACK = 1

turn_right = { LEFT: UP, UP: RIGHT, RIGHT: DOWN, DOWN: LEFT }
turn_left = { turn_right[d] : d for d in turn_right }

def simulate_langtons_ant(steps=11000, rows=100, cols=100, direct=UP):
    row = rows // 2
    col = cols // 2
    plane = np.zeros((rows, cols), dtype=np.int)

    for i in range(steps):
        direct = turn_right[direct] if plane[row, col] == WHITE else turn_left[direct]
        plane[row, col] = not plane[row, col]
        drow, dcol = direct
        row = (row + drow) % rows
        col = (col + dcol) % cols

    return plane

plt.imshow(simulate_langtons_ant())
plt.show()

