"""
implemented based on the "wave function collapse" algorithm by https://github.com/mxgmn/WaveFunctionCollapse
"""
import random
import numpy as np, matplotlib.pyplot as plt

# a direction is a pair (dy, dx) of the change in y and x directions respectively
#   north
# west east
#   south
# n e s w = [(-1, 0), (0, 1), (1, 0), (0, -1)]
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# a tile is a map from directions (the faces of the tile) to logical value
# wherever there is a connection on this side
def nth_tile(n):
    return { (h, v) : bool(n & (1 << i)) for i, (h, v) in enumerate(directions) }

all_tiles = [ nth_tile(n) for n in range(2**4) ]

def opposite_direction(direction):
    return (-direction[0], -direction[1])

def fits_along_direction(tile1, tile2, direction):
    return tile1[direction] == tile2[opposite_direction(direction)]

class WaveFunctionCollapseGrid:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        # set of cell indices (row, col) of cell who are already collapsed to a single tile
        self.collapsed_cells = set()
        # map from cell indices to lists of still possible tiles for this cell index, only for cells which are not yet collapsed
        self.uncollapsed_tiles = { (row, col) : all_tiles for row in range(rows) for col in range(cols) }
        # map from cell indices to the tile to which this cell is collapsed, only for cell which are already collapsed
        self.collapsed_tiles = dict()

    def update_cell(self, cell):
        """
        update this one cell if any neighbors have changed to collapsed and hence pose a constraint on this cell
        """
        y, x = cell
        if cell in self.collapsed_cells:
            return False
        change = False
        for direction in directions:
            dy, dx = direction
            other = (y + dy) % self.rows, (x + dx) % self.cols # wrap edges
            if other in self.collapsed_cells:
                other_tile = self.collapsed_tiles[other]
                new = [ tile for tile in self.uncollapsed_tiles[cell] if fits_along_direction(tile, other_tile, direction) ]
                if len(new) == 0:
                    # reached a contradiction i.e. one possibilities are left for this cell
                    # currently we don't do backtracking to find a solution, the user has to
                    # restart the program with a different seed for the rng
                    raise ValueError("contradiction")
                change = new != self.uncollapsed_tiles[cell]
                self.uncollapsed_tiles[cell] = new
        return change

    def propagate_constraints(self):
        """
        propagate constraints by updating all cells repeatedly until no change occurs anymore
        """
        change = True
        while change:
            change = False
            for row in range(self.rows):
                for col in range(self.cols):
                    change = change or self.update_cell((row, col))

    def find_lowest_entropy_cell(self):
        """
        find the cell with the lowest entropy i.e. the uncollapsed cell with the lowest number of possibilities left open
        """
        lowest_entropy = min(map(len, self.uncollapsed_tiles.values()))
        cells_with_lowest_entropy = [cell for cell, possible_tiles in self.uncollapsed_tiles.items()
                if len(possible_tiles) == lowest_entropy]
        return random.choice(cells_with_lowest_entropy)

    def measure_cell(self, cell):
        """
        measure the given cell: select one of the remaining possibilities for the given (yet uncollapsed cell)
        and mark it as collapsed
        """
        self.collapsed_tiles[cell] = random.choice(self.uncollapsed_tiles[cell])
        del self.uncollapsed_tiles[cell]
        self.collapsed_cells.add(cell)

    def step(self):
        """
        measure the uncollapsed cell with the lowest entropy and propagate constraints
        """
        cell = self.find_lowest_entropy_cell()
        self.measure_cell(cell)
        self.propagate_constraints()

    def collapse(self):
        """
        collapse the complete grid
        """
        while len(self.collapsed_cells) < self.rows * self.cols:
            self.step()

    def plot(self):
        fig, ax = plt.subplots(layout="constrained")
        ax.set_axis_off()
        plt.title("wave function collapse algorithm")
        for cell_index, tile in self.collapsed_tiles.items():
            cell_y, cell_x = cell_index
            for direction in directions:
                if tile[direction]:
                    dy, dx = direction
                    plt.plot([cell_x, cell_x + dx], [cell_y, cell_y + dy], color="black")
            if len([None for direction in directions if tile[direction]]) >= 1:
                plt.plot([cell_x], [cell_y], "or")

if __name__ == "__main__":
    random.seed(42)
    wf = WaveFunctionCollapseGrid(20, 30)
    wf.collapse()
    wf.plot()
    plt.show()

