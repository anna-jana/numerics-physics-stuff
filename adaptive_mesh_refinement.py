from __future__ import annotations

import typing
import dataclasses
import enum
import abc
import logging

import numpy as np
import matplotlib.pyplot as plt

######################################## index arithmentic ###########################################
# <level n y index> <level n x index> <level (n - 1) y index> ... <level 0 x index>

def set_index(index, level, i, j):
    return ((i + 2*j) << (2*level)) | index

def get_index(index, level):
    index >>= 2*level
    return index & 0b01, (index & 0b10) >> 1, index >> 2

def decode_index_to_ints(index, level):
    x_index = y_index = 0
    for l in range(level):
        i, j, _ = get_index(index, l)
        x_index = (x_index << 1) | i
        y_index = (y_index << 1) | j
    return x_index, y_index

def encode_index_from_ints(x_index, y_index, level):
    index = 0
    for l in range(level):
        i = (x_index >> (level - l - 1)) & 1
        j = (y_index >> (level - l - 1)) & 1
        index = set_index(index, l, i, j)
    return index

def add_offset_to_index(index, level, x_offset, y_offset):
    x_index, y_index = decode_index_to_ints(index, level)
    x_index += x_offset
    y_index += y_offset

    if not (0 <= x_index < 2**level and 0 <= y_index < 2**level):
        raise ValueError("index outside of the grid")

    return encode_index_from_ints(x_index, y_index, level)

ROOT_INDEX = 0

######################################## internal node implementation  ########################################
@dataclasses.dataclass
class AMR_TreeNode:
    # spacial and user data
    extend: list[(float, float)] # value range for each dimension
    data: typing.Any
    # tree index math
    level: int
    index_from_root: int
    # pointers down and up
    children: dict[(int, int), AMR_TreeNode]
    parent: None | AMR_TreeNode

    @staticmethod
    def make_root(extend, init_data):
        return AMR_TreeNode(extend, init_data, level=0, index_from_root=ROOT_INDEX,
                children={}, parent=None)

    def has_children(self):
        return len(self.children) != 0

    def add_children(self):
        assert not self.has_children()
        ((x_min, x_max), (y_min, y_max)) = self.extend
        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0
        for i, (x_start, x_end) in enumerate(zip([x_min, x_mid], [x_mid, x_max])):
            for j, (y_start, y_end) in enumerate(zip([y_min, y_mid], [y_mid, y_max])):
                index = set_index(self.index_from_root, self.level, i, j)
                self.children[(i, j)] = AMR_TreeNode(
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

    def plot_node(self, ax, **kwargs):
        ((x_min, x_max), (y_min, y_max)) = self.extend
        ax.plot([x_min, x_max, x_max, x_min, x_min],
                [y_max, y_max, y_min, y_min, y_max],
                **kwargs)

    def plot_tree(self, ax, color):
        if len(self.children) == 0:
            self.plot_node(ax, color=color)
        else:
            for child in self.children.values():
                child.plot_tree(ax, color=color)

    def iter_leafs(self):
        if self.has_children():
            for child in self.children.values():
                yield from child.iter_leafs()
        else:
            yield self

    def iter_all_nodes(self):
        yield self
        for child in self.children.values():
            yield from child.iter_all_nodes()

    def iter_boundary_nodes(self, x_dir, y_dir):
        assert (x_dir == 0) != (y_dir == 0), "one direction has to be zero"
        if not self.has_children():
            yield self
        else:
            # TODO: make this simpler
            if y_dir == 1: # up
                yield from self.children[(0, 1)].iter_boundary_nodes(x_dir, y_dir)
                yield from self.children[(1, 1)].iter_boundary_nodes(x_dir, y_dir)
            elif y_dir == -1: # down
                yield from self.children[(0, 0)].iter_boundary_nodes(x_dir, y_dir)
                yield from self.children[(1, 0)].iter_boundary_nodes(x_dir, y_dir)
            elif x_dir == 1: # right
                yield from self.children[(1, 0)].iter_boundary_nodes(x_dir, y_dir)
                yield from self.children[(1, 1)].iter_boundary_nodes(x_dir, y_dir)
            elif x_dir == -1: # left
                yield from self.children[(0, 0)].iter_boundary_nodes(x_dir, y_dir)
                yield from self.children[(0, 1)].iter_boundary_nodes(x_dir, y_dir)


########################################## user interface ##############################################
class Refinement(enum.Enum):
    KEEP = 0
    REFINE = 1
    COARSE_GRAIN = 2

# abstract base clas
class AMR_Tree(metaclass=abc.ABCMeta):
    def __init__(self, extend, init_data):
        self.root = AMR_TreeNode.make_root(extend, init_data)

    # methods to be defined in the subclass
    @abc.abstractmethod
    def check(self, node): pass

    @abc.abstractmethod
    def refine_data(self, node, index): pass

    @abc.abstractmethod
    def coars_grain_data(self, node): pass

    # tree indexing
    def __getitem__(self, index):
        """
        get node at path index[0] at level index[1]
        """
        if not isinstance(index, tuple):
            raise TypeError(f"(<node path>, <level>) is required as index into tree, {index} given")
        path, target_level = index
        if not isinstance(path, int):
            raise TypeError(f"path should be an int. {path} given")
        if not isinstance(target_level, int):
            raise TypeError(f"target_level should be an int. {target_level} given")
        if target_level < 0:
            raise ValueError(f"target_level should be non-negative, {target_level} given")

        # walk the tree
        remainder = 0 # bring them into scope in case the for loop is never executed (root node)
        current_level = 0
        node = self.root
        while current_level < target_level:
            i, j, remainder = get_index(path, current_level)
            if not node.has_children():
                raise IndexError(f"index {path} does not exits within tree, tree too shallow")
            node = node.children[(i, j)]
            current_level += 1

        if remainder != 0:
            raise IndexError(f"{path = :b} too long for {target_level = }")

        return node

    def get_node_or_above(self, path, target_level):
        """
        get the node at path on level target_level but also allow the tree to be shallower
        and return the leaf node on this path if it ends early
        """
        if not isinstance(path, int):
            raise TypeError(f"path should be an int. {path} given")
        if not isinstance(target_level, int):
            raise TypeError(f"target_level should be an int. {target_level} given")
        if target_level < 0:
            raise ValueError(f"target_level should be non-negative, {target_level} given")

        # walk the tree
        current_level = 0
        node = self.root
        while current_level < target_level:
            i, j, _ = get_index(path, current_level)
            if not node.has_children():
                break
            node = node.children[(i, j)]
            current_level += 1
        return node

    # forward methods to root node
    def iter_leafs(self):
        return self.root.iter_leafs()

    def iter_all_nodes(self):
        return self.root.iter_all_nodes()

    def plot(self, ax, color="black"):
        self.root.plot_tree(ax, color)

    def refine(self, max_levels=None, min_levels=0):
        self.root.refine(self.check, self.refine_data, self.coars_grain_data,
                max_levels, min_levels)

    def get_offset_node(self, node: AMR_TreeNode, x_offset, y_offset):
        path = add_offset_to_index(node.index_from_root, node.level, x_offset, y_offset)
        return self.get_node_or_above(path, node.level)

################################ example: use AMR to approximate 2D function ##############################
class AMR_Function_Tree(AMR_Tree):
    def __init__(self, extend, fn, eps_refine, eps_coarse_grain):
        super().__init__(extend, None)
        self.fn = fn
        self.eps_refine = eps_refine
        self.eps_coarse_grain = eps_coarse_grain

    def check(self, node) -> Refinement:
        ((x_min, x_max), (y_min, y_max)) = node.extend
        mid_linear_approx = (self.fn(x_min, y_min) + self.fn(x_min, y_max) +
                             self.fn(x_max, y_min) + self.fn(x_max, y_max)) / 4.0
        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0
        mid = self.fn(x_mid, y_mid)
        delta = abs(mid_linear_approx - mid)
        if delta >= self.eps_refine:
            return Refinement.REFINE
        elif delta <= self.eps_coarse_grain:
            return Refinement.COARSE_GRAIN
        else:
            return Refinement.KEEP

    def refine_data(self, node, index):
        return None

    def coars_grain_data(self, node):
        return None

########################################## testing ##############################
def test_indexing(tree):
    assert tree[(0, 0)] is tree.root
    for node in tree.iter_all_nodes():
        assert tree[(node.index_from_root, node.level)] is node
    for leaf in tree.iter_leafs():
        assert not leaf.has_children()
    print("tests passed")

def test_plot_adaptation(tree):
    ((x_min, x_max), (y_min, y_max)) = tree.root.extend
    f = tree.fn
    plt.figure()
    ax = plt.gca()
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    cont = ax.contourf(x, y, f(x, y))
    plt.colorbar(cont, ax=ax, label="f(x,y)")
    tree.plot(ax)
    plt.show()

def testing_neighbors(tree):
    plt.figure()
    ax = plt.gca()
    tree.plot(ax)
    all_leafes = list(tree.iter_leafs())
    np.random.seed(1996)

    for i in range(4):
        leaf = np.random.choice(all_leafes)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx == 0) != (dy == 0):
                    try:
                        neighbor = tree.get_offset_node(leaf, dx, dy)
                    except ValueError as ae:
                        if ae.args[0] == "index outside of the grid":
                            print(dx, dy, "neighbor is outside the grid")
                        else:
                            raise
                    neighbor.plot_node(ax, color="red")
        leaf.plot_node(ax, "green")
    plt.show()

def test_boundary(tree):
    plt.figure()
    ax = plt.gca()
    tree.plot(ax)
    all_nodes = list(tree.iter_all_nodes())
    np.random.seed(5)

    while True:
        node = np.random.choice(all_nodes)
        if not node.has_children():
            continue
        dx = -1
        dy = 0
        node.plot_node(ax, color="red", ls="--", lw=2)
        for boundary in node.iter_boundary_nodes(dx, dy):
            boundary.plot_node(ax, color="green", ls="--", lw=2)
            assert not boundary.has_children()
        break
    plt.show()

def make_test_tree():
    extend = ((x_min, x_max), (y_min, y_max)) = [(0, 1.0), (0, 1.0)]
    f = lambda x, y: np.sin(2 * 2*np.pi * np.sqrt((x - 0.5)**2 + (y - 0.5)**2)) / 5.0
    tree = AMR_Function_Tree(extend, f, eps_refine=1e-2, eps_coarse_grain=1e-10)
    tree.refine(max_levels=int(np.log2(100)), min_levels=int(np.log2(5)))
    return tree

if __name__ == "__main__":
    tree = make_test_tree()
    test_boundary(tree)
