import numpy as np, matplotlib.pyplot as plt, itertools

np.random.seed(42)

def step_margolus_ca(rule, nstep, space, next_space):
    nrows_cells, ncols_cells = space.shape
    assert (nrows_cells % 2, ncols_cells % 2) == (0, 0) and space.shape == next_space.shape
    nrows_blocks = nrows_cells // 2
    ncols_blocks = ncols_cells // 2
    offset = nstep % 2
    for row_block in range(nrows_blocks):
        for col_block in range(ncols_blocks):
            row_cell_start = row_block*2 + offset
            col_cell_start = col_block*2 + offset
            N_index = row_cell_start % nrows_cells
            S_index = (row_cell_start + 1) % nrows_cells
            W_index = col_cell_start % ncols_cells
            E_index = (col_cell_start + 1) % ncols_cells
            next_space[N_index, W_index], next_space[N_index, E_index], next_space[S_index, W_index], next_space[S_index, E_index] = \
                rule(space[N_index, W_index], space[N_index, E_index], space[S_index, W_index], space[S_index, E_index])

_hpp_gas_rule_table = {
    (0, 0, 0, 0) : (0, 0, 0, 0),
    (1, 0, 0, 0) : (0, 0, 0, 1),
    (1, 0, 0, 1) : (0, 1, 1, 0),
    (1, 0, 1, 0) : (0, 1, 0, 1),
    (0, 1, 1, 1) : (1, 1, 1, 0),
    (1, 1, 1, 1) : (1, 1, 1, 1),
}

def rot1(block):
    NW, NE, SW, SE = block
    return SW, NW, SE, NE

def flip(block):
    NW, NE, SW, SE = block
    return SW, SE, NW, NE

hpp_gas_rule_table = {}
for rule_entry in _hpp_gas_rule_table.items():
    sym_versions = [rule_entry] # start with rule
    for i in range(3): # add rotated versions
        in_block, out_block = sym_versions[-1]
        sym_versions.append((rot1(in_block), rot1(out_block)))
    # add flipped versions of rotated ones
    sym_versions += [(flip(in_block), flip(out_block)) for in_block, out_block in sym_versions]
    hpp_gas_rule_table.update(dict(sym_versions))

assert len(hpp_gas_rule_table) == 2**4

def hpp_gas_rule(NW, NE, SW, SE):
    return hpp_gas_rule_table[NW, NE, SW, SE]

if __name__ == "__main__":
    N = 100
    space = np.random.randint(0, 2, (N, N))
    space[N//2:, N//2:] = 0
    next_space = np.empty_like(space)
    for nstep in itertools.count():
        plt.clf()
        plt.imshow(space, cmap="Greys")
        plt.pause(0.1)
        step_margolus_ca(hpp_gas_rule, nstep, space, next_space)
        space, next_space = next_space, space

