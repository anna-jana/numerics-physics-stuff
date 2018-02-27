import random
import numpy as np
import matplotlib.pyplot as plt
import pygame

EMPTY = 0
FREE = 1
FIXED = 2

class DLAGrid(object):
    def __init__(self, rows=50, columns=50, free_part=20.0, fixed=[(0,0)]):
        self.grid = np.zeros((rows, columns), dtype=np.int32)

        # put all fixed cell in place
        for fixed_cell_index in fixed:
            self.grid[fixed_cell_index] = FIXED

        # put free (moving) cells at random positions
        total_cell_num = rows*columns
        free_cell_num = int(free_part/100.0*total_cell_num)
        for _ in range(free_cell_num):
            while True:
                row = random.randint(0, rows-1)
                column = random.randint(0, columns-1)
                if self.grid[row, column] == EMPTY:
                    self.grid[row, column] = FREE
                    break

    def plot(self):
        plt.pcolormesh(self.grid)
        plt.show()

    def count_free(self):
        return np.sum(self.grid == FREE)

    def count_fixed(self):
        return np.sum(self.grid == FIXED)

    def get_neighbors(self, row, column):
        for r in (row - 1, row, row + 1):
            for c in (column - 1, column, column + 1):
                if not (r == row and c == column) and 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
                    yield r, c

    def update(self):
        new = np.zeros(self.grid.shape, dtype=np.int32)
        # move the cells in a random order
        rows = np.arange(self.grid.shape[0])
        np.random.shuffle(rows)
        columns = np.arange(self.grid.shape[1])
        np.random.shuffle(columns)
        for row in rows:
            for column in columns:
                cell = self.grid[row, column]
                if cell == EMPTY:
                    continue
                elif cell == FREE:
                    # if any cell around our cell is fixed we become fixed too
                    if any(self.grid[i] == FIXED for i in self.get_neighbors(row, column)):
                        new[row, column] = FIXED
                    # we try to move to some cell around us
                    else:
                        avl_indices = [i for i in self.get_neighbors(row, column)
                                         if self.grid[i] == EMPTY and new[i] == EMPTY]
                        if len(avl_indices) == 0:
                            new[row, column] = FREE
                        else:
                            new[random.choice(avl_indices)] = FREE
                else:
                    # fixed cells stay fixed forever
                    new[row, column] = FIXED
        self.grid = new

    def done(self):
        return self.count_free() == 0

    def steady_state(self):
        while not self.done():
            self.update()

grid = DLAGrid(fixed=[(25, 25)])
grid.steady_state()
grid.plot()
