module GluoDynamics

using Random
using PyPlot
using StaticArrays
using LinearAlgebra
using HDF5
import Base.*

###################### efficient SU3 type ######################
struct SU3
    u::SVector{3,Complex{Float64}}
    v::SVector{3,Complex{Float64}}
end

function get_last_row(U::SU3)
    w = cross(conj(U.u), conj(U.v))
    return w
end

function to_matrix(U::SU3)
    return hcat(U.u, U.v, get_last_row(U))
end

function Base.one(::Type{SU3})
    return SU3(SVector(1, 0, 0),
        SVector(0, 1, 0))
end

function Base.zero(::Type{SU3})
    return SU3(SVector(0, 0, 0),
        SVector(0, 0, 0))
end

# TODO: optimize all the implementations
function LinearAlgebra.inv(U::SU3)
    u, v = U.u, U.v
    w = get_last_row(U)
    u_new = SVector(u[1]', v[1]', w[1]')
    v_new = SVector(u[2]', v[2]', w[2]')
    return SU3(u_new, v_new)
end

function *(U::SU3, V::SU3)
    ans = to_matrix(U) * to_matrix(V)
    return SU3(ans[1,:], ans[2,:])
end

function reproject_su3(U::SU3)::SU3
    u_new = U.u ./ norm(U.u)
    v_prime = U.v - u_new * dot(U.v, u_new)
    return SU3(u_new, v_prime ./ norm(v_prime))
end

const id = [1 0; 0 1]
const pauli = [
    [0 1; 1 0],
    [0 -im; im 0],
    [1 0; 0 -1],
]

function random_su2_to_close_to_1(epsilon)
    r = rand(3) .- 1.0
    x = epsilon * r / norm(r)
    x0 = rand((-1,1)) * sqrt(1 - epsilon^2)
    return x0 * id + sum(im * x[i] * pauli[i] for i in 1:3)
end

function random_su3_close_to_1(epsilon)::SU3
    r = random_su2_to_close_to_1(epsilon)
    s = random_su2_to_close_to_1(epsilon)
    t = random_su2_to_close_to_1(epsilon)
    R = SMatrix{3,3}(r[1,1], r[2,1], 0, r[1,2], r[2,2], 0, 0, 0, 1)
    S = SMatrix{3,3}(s[1,1], 0, s[2,1], 0, 1, 0, s[1,2], 0, s[2,2])
    T = SMatrix{3,3}(1, 0, 0, 0, t[1,1], t[2,1], 0, t[1,2], t[2,2])
    X = R * S * T
    u = X[1, :]
    v = X[2, :]
    return SU3(u, v)
end

######################### simulation type ########################
Base.@kwdef mutable struct GluoDynamicsLattice
    # mc step
    step::Int
    # 4D lattice (1 euclidian time direction, 3 space directions) + 4 link variables per side
    U::Array{SU3,5}
    # neighbors (4D index) for each side in the 4D lattice (2*4 per side)
    stable_indicies::Array{CartesianIndex{5},7}
    # coupling constant
    beta::Float64
    # reproject every nth iterations
    reproject_every::Int
    # possible mc steps to choose from
    choices::Vector{SU3}
    # regenerate possible random choices
    new_choices_every::Int
    # seed of the random number generator
    # the default prng is Xoshiro256++ which has a periode of 2^256 − 1 which is sufficient
    seed::Int
    epsilon::Float64
end

################################# io ############################
function save_simulation_state(filename, s::GluoDynamicsLattice)
    h5open(filename, "w") do f
        f["seed"] = s.seed
        f["U"] = s.U
        f["beta"] = s.beta
        f["reproject_every"] = s.reproject_every
        f["new_choices_every"] = s.new_choices_every
        f["step"] = s.step
        f["n"] = div(length(s.choices), 2)
        f["epsilon"] = s.epsilon
    end
end

function load_simulation_state(filename)::GluoDynamicsLattice
    return h5open(filename, "r") do f
        n = f["n"]
        epsilon = f["epsilon"]
        GluoDynamicsLattice(
            step=f["step"],
            seed=f["seed"],
            U=f["U"],
            beta=f["beta"],
            reproject_every=f["reproject_every"],
            new_choices_every=f["new_choices_every"],
            epsilon=epsilon,
            choices=generate_possible_mc_steps(n, epsilon),
        )
    end
end

############################## links ############################
# index into the link variables (gluon field)
function add_offset!(coord::Vector{Int}, signed_direction::Int, dims::Vector{Int})
    dir = abs(signed_direction)
    sgn = sign(signed_direction)
    coord[dir] = mod1(coord[dir] + sgn, dims[dir])
end

function get_link_index(coord, signed_direction, offsets, dims)
    coord = copy(coord)
    for off in offsets
        add_offset!(coord, off, dims)
    end
    dir = abs(signed_direction)
    if signed_direction < 0
        coord[dir] = mod1(coord[dir] - 1, dims[dir])
    end
    return CartesianIndex(coord..., dir)
end

########################## computing the action ##################
function generate_stable_indicies(N, Nt)
    dims = [N, N, N, Nt]
    stable_indicies = Array{CartesianIndex{5}}(undef, N, N, N, Nt, 4, 3, 6)
    for it in 1:Nt, iz in 1:N, iy in 1:N, ix in 1:N # each lattice location
        n = [ix, iy, iz, it]
        for mu in 1:4 # link direction
            nth = 1
            for nu in 1:4 # other direction of the placett
                if mu != nu
                    stable_indicies[n..., mu, nth, 1] = get_link_index(n, nu, [mu], dims)
                    stable_indicies[n..., mu, nth, 2] = get_link_index(n, -mu, [nu, mu], dims)
                    stable_indicies[n..., mu, nth, 3] = get_link_index(n, -nu, [nu], dims)
                    stable_indicies[n..., mu, nth, 4] = get_link_index(n, -nu, [mu], dims)
                    stable_indicies[n..., mu, nth, 5] = get_link_index(n, -mu, [-nu, mu], dims)
                    stable_indicies[n..., mu, nth, 6] = get_link_index(n, nu, [-nu], dims)
                    nth += 1
                end
            end
        end
    end
    return stable_indicies
end

# compute the action difference when changing one link
function compute_stable(s::GluoDynamicsLattice, i::CartesianIndex{5})
    A = zeros(Complex{Float64}, 3, 3)
    mu = i[length(i)]
    nu_index = 1
    for nu in 1:4
        if nu != mu
            A += to_matrix(s.U[s.stable_indicies[i, nu_index, 1]] *
                           s.U[s.stable_indicies[i, nu_index, 2]] *
                           s.U[s.stable_indicies[i, nu_index, 3]])
            A += to_matrix(s.U[s.stable_indicies[i, nu_index, 4]] *
                           s.U[s.stable_indicies[i, nu_index, 5]] *
                           s.U[s.stable_indicies[i, nu_index, 6]])
            nu_index += 1
        end
    end
    return A
end

function calc_action_diff(s::GluoDynamicsLattice, i::CartesianIndex{5}, U_new::SU3)
    N = size(s.U, 1)
    A = compute_stable(s, i)
    U = to_matrix(s.U[i])
    U_new_matrix = to_matrix(U_new)
    # TODO: make this more efficient
    return - s.beta / N * real(tr((U_new_matrix - U) * A))
end

############################ initialization ########################
# cold start = low temp = 1 on each link
function cold_start(N, Nt)
    return ones(SU3, (N, N, N, Nt, 4))
end

# generate mc steps
function generate_possible_mc_steps(n, epsilon)::Vector{SU3}
    choices_half = [random_su3_close_to_1(epsilon) for _ in 1:n]
    inv_half = inv.(choices_half)
    return vcat(choices_half, inv_half)
end

function GluoDynamicsLattice(s, N, Nt, nchoices, beta, reproject_every, new_choices_every, epsilon)
    Random.seed!(s)
    return GluoDynamicsLattice(
        step=0,
        U=cold_start(N, Nt),
        stable_indicies=generate_stable_indicies(N, Nt),
        beta=beta,
        reproject_every=reproject_every,
        choices=generate_possible_mc_steps(nchoices, epsilon),
        new_choices_every=new_choices_every,
        seed=s,
        epsilon=epsilon,
    )
end

######################### mc algorithm ########################
function mc_step!(s::GluoDynamicsLattice, i::CartesianIndex{5})
    # metropolis for a first test
    X = rand(s.choices)
    U_new = X * s.U[i]
    Delta_S = calc_action_diff(s, i, U_new)
    p = rand()
    if p < exp(-Delta_S)
        s.U[i] = U_new
    end
end

function mc_sweep!(s::GluoDynamicsLattice)
    for i in CartesianIndices(s.U)
        mc_step!(s, i)
    end
end

######################### observables ########################
function eval_polyakov_loop(s::GluoDynamicsLattice)
    # TODO
end

function eval_all_polyakov_loops(s::GluoDynamicsLattice)
    # TODO
end

##################### the main mc loop #########################
function main_loop!(s::GluoDynamicsLattice, nsteps, evaluate)
    dataseries = []
    for step in nsteps
        if step % s.reproject_every == 0
            map!(reproject_su3, s.U, s.U)
        end
        if step % s.new_choices_every == 0
            s.choices = generate_possible_mc_steps(div(length(s.choices), 2),
                                                   s.epsilon)
        end
        mc_sweep!(s)
        if evaluate
            data = eval_all_polyakov_loops(s)
            push!(dataseries, (step, data))
        end
    end
    return dataseries
end

end


function test()
    gdl = GluoDynamics.GluoDynamicsLattice(42, 10, 10, 1000, 1.0, 100, 100, 0.01)
    GluoDynamics.main_loop!(gdl, 10^3, false)
end
