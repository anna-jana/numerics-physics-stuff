# https://www.youtube.com/playlist?list=PLnJ8lIgfDbkoZ33CHr-p6z2CBkp9OTcWj
# Numerical Recipes 3rd edition - chapter 20.6
#
using Statistics
using PyPlot
using LinearAlgebra

const T_R = 1.0
const T_L = 2.0
const T_U = 3.0
const T_B = 4.0
const Lx = 1.0
const Ly = 2.0
const eps = 1e-10

function gauss_seidel_sweep!(grid, lattice_constant, rhs)
    (dx, dy) = lattice_constant
    prefactor = -2*(1/dx^2 + 1/dy^2)
    for j in 2:size(grid, 2) - 1
        for i in 2:size(grid, 1) - 1
            neighbor_sum = (grid[i - 1, j] + grid[i + 1, j]) / dx^2 + (grid[i, j - 1] + grid[i, j + 1]) / dy^2
            grid[i, j] = (rhs[i, j] - neighbor_sum) / prefactor
        end
    end
end

function residual!(r, grid, lattice_constant, rhs)
    (dx, dy) = lattice_constant
    for j in 2:size(grid, 2) - 1
        for i in 2:size(grid, 1) - 1
            lhs = (grid[i - 1, j] - 2*grid[i, j] + grid[i + 1, j]) / dx^2 + (grid[i, j - 1] - 2*grid[i, j] + grid[i, j + 1]) / dy^2
            r[i, j] = rhs[i, j] - lhs
        end
    end
end

function restriction!(coarser_grid, finer_grid, block_size)
    (Nx_coarser, Ny_coarser) = size(coarser_grid)
    (Nx_finer, Ny_finer) = size(finer_grid)
    for j in 2:Ny_coarser - 1
        for i in 2:Nx_coarser - 1
            # one 1 from one-based indexing, one 1 from the 1 offset from the boundary
            a = block_size*(i - 1 - 1) + 1 + 1
            b = block_size*(j - 1 - 1) + 1 + 1
            coarser_grid[i, j] = 0.0
            for b_off in 0:block_size-1
                for a_off in 0:block_size-1
                    coarser_grid[i, j] += finer_grid[a + a_off, b + b_off]
                end
                coarser_grid[i, j] /= block_size^2
            end
        end
    end
end

function prolongation!(finer_grid, coarser_grid, block_size)
    (Nx_coarser, Ny_coarser) = size(coarser_grid)
    (Nx_finer, Ny_finer) = size(finer_grid)
    for j in 2:Ny_coarser - 1
        for i in 2:Ny_coarser - 1
            a = block_size*(i - 1 - 1) + 1 + 1
            b = block_size*(j - 1 - 1) + 1 + 1
            for b_off in 0:block_size-1
                for a_off in 0:block_size-1
                    finer_grid[a + a_off, b + b_off] = coarser_grid[i, j]
                end
            end
        end
    end
end

@enum MultigridDirection begin
    up # to coarser grid
    down # to finer grid
end

# for performance comparison
function gauss_seidel(Nx, Ny)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    T = zeros(Nx, Ny)
    T[:, 1] .= T_U
    T[:, end] .= T_B
    T[1, :] .= T_L
    T[end, :] .= T_R

    rhs = zeros(Nx, Ny)
    residual = zeros(Nx, Ny)

    iteration = 0
    while true
        gauss_seidel_sweep!(T, (dx, dy), rhs)
        residual!(residual, T, (dx, dy), rhs)
        mse = sqrt(mean(abs2.(residual)))
        iteration += 1
        if mse < eps
            @show iteration
            return T
        end
    end
end

function multi_grid(fine_Nx, fine_Ny)
    # solver (discretisation) parameters
    nlevels = 4 # how many time are we going to a courser grid
    n_remove_high_frequency_sweeps = 10 # max(fine_Nx, fine_Ny) # 10
    n_remove_low_frequency_sweeps= 10 # n_remove_high_frequency_sweeps # 10
    n_finalize_sweeps = 10
    block_size = 2
    multigrid_directions = vcat(repeat([up], nlevels - 1), repeat([down], nlevels - 1)) # V-cycle

    grid_sizes = [(fine_Nx, fine_Ny)]
    for i in 2:nlevels
        Nx, Ny = grid_sizes[end]
        @assert (Nx - 2) % block_size == 0 && (Ny - 2) % block_size == 0
        push!(grid_sizes, (div(Nx - 2, block_size) + 2, div(Ny - 2, block_size) + 2))
    end
    reverse!(grid_sizes)

    lattice_constants = [(Lx / (Nx_coarse - 1), Ly / (Ny_coarse - 1)) for (Nx_coarse, Ny_coarse) in grid_sizes]

    grids = map(zeros, grid_sizes)
    grids[end][:, 1] .= T_U
    grids[end][:, end] .= T_B
    grids[end][1, :] .= T_L
    grids[end][end, :] .= T_R

    rhss = map(zeros, grid_sizes)
    # fill!(rhss[end], 0.0) # set the rhs for the original equation
    Delta_xs = map(zeros, grid_sizes)
    residuals = map(zeros, grid_sizes)
    grid_index = nlevels # start with the finest grid

    # how to move the matrix A:
    #   - algebraic multigrid: A' = RAP
    #   - geometric multigrid: reconstruct A for the new grid size, normally more expensive and memory consuming but
    #       in our case we dont want to explicitly construct the matrix A at all
    #       and only define how to apply A and to apply a gauss seidel sweep
    #       hence A is the same except for changing lattice constants dx and dy
    iteration = 0
    while true
        for multigrid_direction in multigrid_directions
            if multigrid_direction == up # to coarser grid
                # remove high frequency errors
                for i in 1:n_remove_high_frequency_sweeps
                    gauss_seidel_sweep!(grids[grid_index], lattice_constants[grid_index], rhss[grid_index])
                end
                # compute residual r = b - Ax
                residual!(residuals[grid_index], grids[grid_index], lattice_constants[grid_index], rhss[grid_index])
                # restriction of residual to coarser grid -> rhs for coarser eq. (which is possible bc residual is smooth
                # (as it is an approximation to the iteration error) (the actual solution might not be smooth))
                restriction!(rhss[grid_index - 1], residuals[grid_index], block_size) # new rhs is the residual
                grid_index -= 1
            elseif multigrid_direction == down # to finer grid
                # remove high frequency errors in coarser grid (i.e. low frequency errors in finer grid) of the eq. A Delta x = r
                for i in 1:n_remove_low_frequency_sweeps
                    gauss_seidel_sweep!(grids[grid_index], lattice_constants[grid_index], rhss[grid_index])
                end
                # move Delta x to finer grid
                prolongation!(Delta_xs[grid_index + 1], grids[grid_index], block_size)
                grid_index += 1
                # update x with Delta x
                grids[grid_index] .+= Delta_xs[grid_index]
                # remove errors from prolongation
                for i in 1:n_finalize_sweeps
                    gauss_seidel_sweep!(grids[grid_index], lattice_constants[grid_index], rhss[grid_index])
                end
            else
                error("invalid multigrid_direction $multigrid_direction")
            end
        end
        @assert grid_index == nlevels
        residual!(residuals[end], grids[end], lattice_constants[end], rhss[end])
        res = sqrt(mean(abs2.(residuals[end])))
        iteration += 1
        if res < eps
            @show iteration*(n_remove_high_frequency_sweeps + n_finalize_sweeps)
            return grids[end]
        end
    end
end

function test()
    Nx = 2 + 2^6
    Ny = 2 + 2^6
    @show Nx, Ny
    println("gauss seidel")
    @time gauss_seidel(Nx, Ny)
    println("multigrid")
    @time multi_grid(Nx, Ny)
    nothing
end
test()
