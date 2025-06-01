from __future__ import annotations
from typing import Any
import dataclasses
import enum
import logging
import numpy as np
import matplotlib.pyplot as plt

class Refinement(enum.Enum):
    KEEP = 0
    REFINE = 1
    COARSE_GRAIN = 2

@dataclasses.dataclass
class AMR_Tree:
    # spacial and user data
    extend: list[(float, float)] # value range for each dimension
    data: Any
    # tree index math
    level: int
    index_from_root: int
    # pointers down and up
    children: dict[(int, int), AMR_Tree]
    parent: None | AMR_Tree

    @staticmethod
    def make_root(extend, init_data=None):
        return AMR_Tree(extend, init_data, level=0, index_from_root=0,
                children={}, parent=None)

    @staticmethod
    def construct(extend, check, refine_data, coarse_grain_data,
            max_levels=None, min_levels=0, init_data=None):
        tree = AMR_Tree.make_root(extend, init_data=init_data)
        tree.refine(check, refine_data, coarse_grain_data,
                max_levels=max_levels, min_levels=min_levels)
        return tree

    def has_children(self):
        return len(self.children) != 0

    def add_children(self):
        assert not self.has_children()
        ((x_min, x_max), (y_min, y_max)) = self.extend
        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0
        for i, (x_start, x_end) in enumerate(zip([x_min, x_mid], [x_mid, x_max])):
            for j, (y_start, y_end) in enumerate(zip([y_min, y_mid], [y_mid, y_max])):
                # create numeric index using the floowing bit pattern:
                # <level n y index> <level n x index> <level (n - 1) y index> ... <level 0 x index>
                index = ((i + 2*j) << (2*self.level)) | self.index_from_root
                self.children[(i, j)] = AMR_Tree(
                        extend=[(x_start, x_end), (y_start, y_end)],
                        data=None,
                        level=self.level + 1,
                        index_from_root=index,
                        children={},
                        parent=self
                )

    def remove_children(self):
        assert self.has_children()
        for child in self.children.values():
            child.parent = None
        self.children = {}

    def refine(self, check, refine_data, coarse_grain_data, max_levels=None, min_levels=0):
        match check(self):
            case Refinement.KEEP:
                return
            case Refinement.REFINE:
                if max_levels is not None and self.level >= max_levels:
                    # check if max level of subdivision is reached
                    logging.warn("max levels reached but need to refine")
                else:
                    # extend the data structure
                    self.add_children()
                    # refine the data from our tree
                    for index, child in self.children.items():
                        child.data = refine_data(self, index)
                    # recursive refining
                    for child in self.children.values():
                        child.refine(check, refine_data, coarse_grain_data,
                                max_levels=max_levels, min_levels=min_levels)
            case Refinement.COARSE_GRAIN:
                if self.level >= min_levels and self.has_children():
                    self.data = coarse_grain_data(self)
                    self.remove_children()

    def plot(self, ax, color="red"):
        if len(self.children) == 0:
            ((x_min, x_max), (y_min, y_max)) = self.extend
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_max, y_max, y_min, y_min, y_max],
                    color=color)
        else:
            for child in self.children.values():
                child.plot(ax, color=color)

    def __iter__(self):
        if self.has_children():
            for child in self.children.values():
                yield from iter(child) # recursive call
        else:
            yield self

    def __getitem__(self, path):
        if isinstance(path, list):
            if len(path) == 0:
                return self
            if not self.has_children():
                raise IndexError(f"tree ends on level {self.level} but index path {path} remains")
            if path[0] not in self.children:
                raise IndexError(f"invalid tree index {path[0]}")
            child = self.children[path[0]]
            return child[path[1:]] # recursive call
        elif isinstance(path, int):
            if not self.has_children():
                raise IndexError(f"indexing leaf tree with {path}")
            if path == 0:
                return self
            # unpack bit pattern (documented in refine method)
            i = path & 1
            j = (path & (1 << 1)) >> 1
            child = self.children[(i, j)]
            return child[path >> 2]

    def get_neighboring_block(self, offsets):
        pass


# use AMR to approximate 2D function
def make_check_function(fn, eps_refine, eps_coarse_grain):
    def check(tree) -> Refinement:
        ((x_min, x_max), (y_min, y_max)) = tree.extend
        mid_linear_approx = (fn(x_min, y_min) + fn(x_min, y_max) +
                             fn(x_max, y_min) + fn(x_max, y_max)) / 4.0
        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0
        mid = fn(x_mid, y_mid)
        delta = abs(mid_linear_approx - mid)
        if delta >= eps_refine:
            return Refinement.REFINE
        elif delta <= eps_coarse_grain:
            return Refinement.COARSE_GRAIN
        else:
            return Refinement.KEEP
    return check

def refine_function(tree, index):
    return None

def coarse_grain_function(tree):
    return None

if __name__ == "__main__":
    # create tree
    extend = ((x_min, x_max), (y_min, y_max)) = [(0, 1.0), (0, 1.0)]
    f = lambda x, y: np.sin(2 * 2*np.pi * np.sqrt((x - 0.5)**2 + (y - 0.5)**2)) / 5.0
    tree = AMR_Tree.construct(extend,
            make_check_function(f, eps_refine=1e-2, eps_coarse_grain=1e-10),
            refine_function, coarse_grain_function,
            max_levels=int(np.log2(100)), min_levels=int(np.log2(5)))

    assert tree[0b1011].index_from_root == 0b1011

    # plotting
    fig = plt.figure()
    ax = plt.gca()
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    cont = ax.contourf(x, y, f(x, y))
    tree.plot(ax)
    plt.colorbar(cont, ax=ax)
    plt.show()
