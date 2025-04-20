# https://www.youtube.com/playlist?list=PLnJ8lIgfDbkoZ33CHr-p6z2CBkp9OTcWj
# Numerical Recipes 3rd edition - chapter 20.6

import numpy as np
import numba
import time
import enum

@numba.njit
def gauss_seidel_sweep(grid, dx, dy, rhs):
    prefactor = -2*(1/dx**2 + 1/dy**2)
    Nx, Ny = grid.shape
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            neighbor_sum = (grid[i - 1, j] + grid[i + 1, j]) / dx**2 + (grid[i, j - 1] + grid[i, j + 1]) / dy**2
            grid[i, j] = (rhs[i, j] - neighbor_sum) / prefactor

@numba.njit
def compute_residual(r, grid, dx, dy, rhs):
    Nx, Ny = grid.shape
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            lhs = (grid[i - 1, j] - 2*grid[i, j] + grid[i + 1, j]) / dx**2 + (grid[i, j - 1] - 2*grid[i, j] + grid[i, j + 1]) / dy**2
            r[i, j] = rhs[i, j] - lhs

@numba.njit
def restriction(coarser_grid, finer_grid, block_size):
    Nx_coarser, Ny_coarser = coarser_grid.shape
    Nx_finer, Ny_finer = finer_grid.shape
    for i in range(1, Nx_coarser - 1):
        for j in range(1, Ny_coarser - 1):
            a = block_size*(i - 1) + 1
            b = block_size*(j - 1) + 1
            coarser_grid[i, j] = 0.0
            for b_off in range(block_size):
                for a_off in range(block_size):
                    coarser_grid[i, j] += finer_grid[a + a_off, b + b_off]
                coarser_grid[i, j] /= block_size**2

@numba.njit
def prolongation(finer_grid, coarser_grid, block_size):
    Nx_coarser, Ny_coarser = coarser_grid.shape
    Nx_finer, Ny_finer = finer_grid.shape
    for i in range(1, Nx_coarser - 1):
        for j in range(1, Ny_coarser - 1):
            a_base = block_size*(i - 1) + 1
            b_base = block_size*(j - 1) + 1
            for b_off in range(block_size):
                for a_off in range(block_size):
                    finer_grid[a_base + a_off, b_base + b_off] = coarser_grid[i, j]

class MultigridDirection(enum.Enum):
    up = 0 # to coarser grid
    down = 1 # to finer grid

# for performance comparison
def gauss_seidel(grid, dx, dy, rhs):
    residual = np.zeros((Nx, Ny))
    iteration = 0
    while True:
        gauss_seidel_sweep(T, dx, dy, rhs)
        compute_residual(residual, T, dx, dy, rhs)
        mse = np.sqrt(np.mean(residual**2))
        iteration += 1
        if mse < eps:
            print("iterations:", iteration)
            return T

def multi_grid(grid, dx, dy, rhs):
    fine_Nx, fine_Ny = grid.shape
    # solver (discretisation) parameters
    nlevels = 4 # how many time are we going to a courser grid
    n_remove_high_frequency_sweeps = 10 # max(fine_Nx, fine_Ny) # 10
    n_remove_low_frequency_sweeps= 10 # n_remove_high_frequency_sweeps # 10
    n_finalize_sweeps = 10
    block_size = 2
    multigrid_directions = [MultigridDirection.up] * (nlevels - 1) + [MultigridDirection.down] * (nlevels - 1) # V-cycle

    grid_sizes = [(fine_Nx, fine_Ny)]
    for i in range(2, nlevels + 1):
        Nx, Ny = grid_sizes[-1]
        assert (Nx - 2) % block_size == 0 and (Ny - 2) % block_size == 0
        grid_sizes.append(((Nx - 2) // block_size + 2, (Ny - 2) // block_size + 2))
    grid_sizes = list(reversed(grid_sizes))

    lattice_constants = [(Lx / (Nx_coarse - 1), Ly / (Ny_coarse - 1)) for Nx_coarse, Ny_coarse in grid_sizes]

    grids = list(map(np.zeros, grid_sizes[:-1]))
    grids.append(grid)

    rhss = list(map(np.zeros, grid_sizes[:-1]))
    rhss.append(rhs)
    # fill(rhss[-1], 0.0) # set the rhs for the original equation
    Delta_xs = list(map(np.zeros, grid_sizes))
    residuals = list(map(np.zeros, grid_sizes))
    grid_index = nlevels - 1 # start with the finest grid

    # how to move the matrix A:
    #   - algebraic multigrid: A' = RAP
    #   - geometric multigrid: reconstruct A for the new grid size, normally more expensive and memory consuming but
    #       in our case we dont want to explicitly construct the matrix A at all
    #       and only define how to apply A and to apply a gauss seidel sweep
    #       hence A is the same except for changing lattice constants dx and dy
    iteration = 0
    while True:
        for multigrid_direction in multigrid_directions:
            if multigrid_direction == MultigridDirection.up: # to coarser grid
                # remove high frequency errors
                for i in range(n_remove_high_frequency_sweeps):
                    gauss_seidel_sweep(grids[grid_index], *lattice_constants[grid_index], rhss[grid_index])

                # compute residual r = b - Ax
                compute_residual(residuals[grid_index], grids[grid_index], *lattice_constants[grid_index], rhss[grid_index])
                # restriction of residual to coarser grid -> rhs for coarser eq. (which is possible bc residual is smooth
                # (as it is an approximation to the iteration error) (the actual solution might not be smooth))
                restriction(rhss[grid_index - 1], residuals[grid_index], block_size) # new rhs is the residual
                grid_index -= 1
            elif multigrid_direction == MultigridDirection.down: # to finer grid
                # remove high frequency errors in coarser grid (i.e. low frequency errors in finer grid) of the eq. A Delta x = r
                for i in range(n_remove_low_frequency_sweeps):
                    gauss_seidel_sweep(grids[grid_index], *lattice_constants[grid_index], rhss[grid_index])

                # move Delta x to finer grid
                prolongation(Delta_xs[grid_index + 1], grids[grid_index], block_size)
                grid_index += 1
                # update x with Delta x
                grids[grid_index] += Delta_xs[grid_index]
                # remove errors from prolongation
                for i in range(n_finalize_sweeps):
                    gauss_seidel_sweep(grids[grid_index], *lattice_constants[grid_index], rhss[grid_index])

            else:
                raise ValueError(f"invalid multigrid_direction {multigrid_direction}")

        assert grid_index == nlevels - 1
        compute_residual(residuals[-1], grids[-1], *lattice_constants[-1], rhss[-1])
        res = np.sqrt(np.mean((residuals[-1])**2))
        iteration += 1
        if res < eps:
            print("iterations:", iteration*(n_remove_high_frequency_sweeps + n_finalize_sweeps))
            return grids[-1]

if __name__ == "__main__":
    T_R = 1.0
    T_L = 2.0
    T_U = 3.0
    T_B = 4.0
    Lx = 1.0
    Ly = 2.0
    eps = 1e-10
    Nx = 2 + 2**6
    Ny = 2 + 2**6
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    T = np.zeros((Nx, Ny))
    T[:, 0] = T_U
    T[:, -1] = T_B
    T[0, :] = T_L
    T[-1, :] = T_R
    rhs = np.zeros((Nx, Ny))
    print("gauss seidel")
    start = time.perf_counter()
    gauss_seidel(T.copy(), dx, dy, rhs.copy())
    print("time:", time.perf_counter() - start)
    print("multigrid")
    start = time.perf_counter()
    multi_grid(T.copy(), dx, dy, rhs.copy())
    print("time:", time.perf_counter() - start)
