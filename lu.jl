importall Base

"""
storage for a LU decomposition PA = LU where P is stored as an index vector
and LU is stored in one matrix
"""
immutable LU{T <: Real}
    LU::Matrix{T}
    permuation::Vector{Int}
end

"""
LU decomposition with colunm pivot selection PA = LU of a given matrix A
using the doolittle algorithm (~ gaussian elimination)
"""
function LU{T <: Real}(A::Matrix{T})
    @assert size(A, 1) == size(A, 2)
    n = size(A, 1)
    lu = copy(A)
    permuation = collect(1:n)
    for row_index in 1:n
        perm_row_index = permuation[row_index]

        # select pivot
        swap_index = row_index
        max_pivot = lu[perm_row_index, row_index]
        for row_to_swap_index in row_index:n
            perm_row_to_swap_index = permuation[row_to_swap_index]
            pivot = lu[perm_row_to_swap_index, row_index]
            if abs(pivot) > abs(max_pivot)
                max_pivot = pivot
                swap_index = row_to_swap_index
            end
        end

        # swap rows
        permuation[row_index], permuation[swap_index] = permuation[swap_index], permuation[row_index]
        perm_row_index = permuation[row_index]

        # elimination
        # eliminate in each row
        for row_to_elim_index in row_index+1:n
            perm_row_to_elim_index = permuation[row_to_elim_index]
            elimination_factor = lu[perm_row_to_elim_index, row_index]/max_pivot
            # perform the elimination on the row (only on the upper part)
            for var_col_index in row_index:n
                lu[perm_row_to_elim_index, var_col_index] -= elimination_factor*lu[perm_row_index, var_col_index]
            end
            # store the factor (the matrix inverse) in the lower part
            lu[perm_row_to_elim_index, row_index] = elimination_factor
        end
    end
    return LU{T}(lu, permuation)
end

"""
solves a linear system Ax = b using a given LU decomposition PA = LU
PAx = LUx = b
Ly = b # forward
Ux = y # backward
"""
function solve{T <: Real}(lu::LU{T}, b::Vector{T})
    n = size(lu.permuation, 1)

    y = Array(T, n)
    for row_index in 1:n
        perm_row_index = lu.permuation[row_index]
        s = 0
        for col_index in 1:row_index-1
            s += lu.LU[perm_row_index, col_index]*y[col_index]
        end
        y[row_index] = b[perm_row_index] - s
    end

    x = Array(T, n)
    for row_index in n:-1:1
        perm_row_index = lu.permuation[row_index]
        s = 0
        for col_index in row_index+1:n
            s += lu.LU[perm_row_index, col_index]*x[col_index]
        end
        x[row_index] = (y[row_index] - s)/lu.LU[perm_row_index, row_index]
    end

    return x
end

A = [1. 2. 3.
     1. 1. 1.
     3. 3. 1.]

b = [1., 8., 7.]

function test()
    println("solution", A\b)
    my = LU(A) # ok
    println(solve(my, b))
end
