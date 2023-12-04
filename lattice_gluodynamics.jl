module GluoDynamics

using Random
using PyPlot
using StaticArrays
using LinearAlgebra
import Base.*
using JLD2

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

function LinearAlgebra.tr(U::SU3)
    w = get_last_row(U)
    return U.u[1] + U.v[2] + w[3]
end

function *(U::SU3, V::SU3)
    ans = to_matrix(U) * to_matrix(V)
    return SU3(ans[1, :], ans[2, :])
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

function random_su2_to_close_to_1(rng, epsilon)
    r = rand(rng, 3) .- 1.0
    x = epsilon * r / norm(r)
    x0 = rand(rng, (-1, 1)) * sqrt(1 - epsilon^2)
    return x0 * id + sum(im * x[i] * pauli[i] for i in 1:3)
end

function random_su3_close_to_1(rng, epsilon)::SU3
    r = random_su2_to_close_to_1(rng, epsilon)
    s = random_su2_to_close_to_1(rng, epsilon)
    t = random_su2_to_close_to_1(rng, epsilon)
    R = SMatrix{3,3}(r[1, 1], r[2, 1], 0, r[1, 2], r[2, 2], 0, 0, 0, 1)
    S = SMatrix{3,3}(s[1, 1], 0, s[2, 1], 0, 1, 0, s[1, 2], 0, s[2, 2])
    T = SMatrix{3,3}(1, 0, 0, 0, t[1, 1], t[2, 1], 0, t[1, 2], t[2, 2])
    X = R * S * T
    u = X[1, :]
    v = X[2, :]
    return SU3(u, v)
end

######################### simulation type ########################
Base.@kwdef mutable struct GluoDynamicsLattice
    # size of the time direction
    Nt::Int
    # size of the space directions
    N::Int
    # mc step
    step::Int
    # 4D lattice (1 euclidian time direction, 3 space directions) + 4 link variables per side
    U::Array{SU3,5}
    # neighbors (4D index) for each side in the 4D lattice (2*4 per side)
    stable_indicies::Array{CartesianIndex{5},7}
    # inverse coupling constant
    beta::Float64
    # reproject every nth iterations
    reproject_every::Int
    # possible mc steps to choose from
    nchoices::Int
    choices::Vector{SU3}
    # regenerate possible random choices
    new_choices_every::Int
    # the prng is Xoshiro256++ which has a periode of 2^256 − 1 which is sufficient
    rng::Xoshiro
    # small number for generation of mc step proposals (which should be close to unity)
    epsilon::Float64
end

############################## links ############################
# index into the link variables (gluon field)
function add_offset!(coord::Vector{Int}, signed_direction::Int, dims::Vector{Int})
    dir = abs(signed_direction)
    sgn = sign(signed_direction)
    coord[dir] = mod1(coord[dir] + sgn, dims[dir])
end

function get_link_index(coord, signed_direction, offsets, dims)
    coord = collect(Tuple(coord)) # CartesianIndex -> Array
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
    @info "setting up stable indicies"
    dims = [N, N, N, Nt]
    stable_indicies = Array{CartesianIndex{5}}(undef, N, N, N, Nt, 4, 3, 6)
    for n in CartesianIndices((N, N, N, Nt)) # each lattice location
        for mu in 1:4 # link direction
            nth = 1
            for nu in 1:4 # other direction of the placett
                if mu != nu
                    stable_indicies[n, mu, nth, 1] = get_link_index(n, nu, [mu], dims)
                    stable_indicies[n, mu, nth, 2] = get_link_index(n, -mu, [nu, mu], dims)
                    stable_indicies[n, mu, nth, 3] = get_link_index(n, -nu, [nu], dims)
                    stable_indicies[n, mu, nth, 4] = get_link_index(n, -nu, [mu], dims)
                    stable_indicies[n, mu, nth, 5] = get_link_index(n, -mu, [-nu, mu], dims)
                    stable_indicies[n, mu, nth, 6] = get_link_index(n, nu, [-nu], dims)
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
    return -s.beta / N * real(tr((U_new_matrix - U) * A))
end

############################ initialization ########################
# cold start = low temp = 1 on each link
function cold_start(N, Nt)
    return ones(SU3, (N, N, N, Nt, 4))
end

# generate mc steps
function generate_possible_mc_steps(rng, n, epsilon)::Vector{SU3}
    choices_half = [random_su3_close_to_1(rng, epsilon) for _ in 1:n]
    inv_half = inv.(choices_half)
    return vcat(choices_half, inv_half)
end

function GluoDynamicsLattice(seed, N, Nt, nchoices, beta, reproject_every, new_choices_every, epsilon)
    @info "building new lattice simulation $N^3*$Nt with seed $seed @ beta = $beta"
    rng = Xoshiro(seed)
    return GluoDynamicsLattice(
        Nt=Nt,
        N=N,
        step=0,
        U=cold_start(N, Nt),
        stable_indicies=generate_stable_indicies(N, Nt),
        beta=beta,
        reproject_every=reproject_every,
        nchoices=nchoices,
        choices=generate_possible_mc_steps(rng, nchoices, epsilon),
        new_choices_every=new_choices_every,
        rng=rng,
        epsilon=epsilon,
    )
end

######################### mc algorithm ########################
function mc_step!(s::GluoDynamicsLattice, i::CartesianIndex{5})
    # metropolis for a first test
    # use a random step
    X = rand(s.choices)
    U_new = X * s.U[i]
    # compute action difference for acceptance propability
    Delta_S = calc_action_diff(s, i, U_new)
    # accept
    p = rand()
    if p < exp(-Delta_S)
        s.U[i] = U_new
    end
end

function mc_sweep!(s::GluoDynamicsLattice)
    for n in CartesianIndices(s.U)
        mc_step!(s, n)
    end
    s.step += 1
end

######################### observables ########################
function eval_polyakov_loop(s::GluoDynamicsLattice, n::CartesianIndex{3})
    return tr(prod(s.U[n, nt, 4] for nt in 1:s.Nt))
end

function compute_polykov_correlator(s::GluoDynamicsLattice)
    @info "polyako loops"
    # eval each polykov loop
    P = [eval_polyakov_loop(s, n) for n in CartesianIndices((s.N, s.N, s.N))]
    # compute their correlator as a histogram
    # only the distance between the points matters bc of translation invariance
    hist = Dict{Int, Complex{Float64}}()
    counts = Dict{Int, Int}()
    for n in CartesianIndices(P)
        for m in CartesianIndices(P)
            d2 = (n[1] - m[1])^2 +  (n[2] - m[2])^2 + (n[3] - m[3])^2
            if !haskey(hist, d2)
                hist[d2] = 0.0 + 0.0im
                counts[d2] = 0
            end
            hist[d2] += P[n] * P[m]'
            counts[d2] += 1
        end
    end
    return hist
end

##################### the main mc loop #########################
function load_simulation_state(filename)::GluoDynamicsLattice
    @info "loading simulation state from $filename"
    return load_object(filename)
end

function main_loop!(s::GluoDynamicsLattice, nsteps, evaluate, eval_every; info_every_secs = 1)
    @info "running simulation"
    filename_polykov = "polyakov_loops.jld"
    @info "saving dataseries of polykov loops in $filename_polykov"
    jldopen(filename_polykov, "w") do f
        # last time we printed loop update (step) info
        last_info = time()
        # main mc loop
        for step in 1:nsteps
            now = time()
            if abs(last_info - now) > info_every_secs
                # use the step of this run, so we can watch the progress
                @info "step = $step / $nsteps"
                # restart timer for step printing
                last_info = now
            end
            # here we use the total step (for reproducablity)
            # make sure that the su3 field stays in su3
            if s.step % s.reproject_every == 0
                map!(reproject_su3, s.U, s.U)
            end
            # here we use the total step (for reproducablity)
            # new possible mc steps
            if s.step % s.new_choices_every == 0
                s.choices = generate_possible_mc_steps(s.rng, s.nchoices, s.epsilon)
            end
            # apply one mc sweep (touch all lattcie sides/links)
            mc_sweep!(s)
            # here we use the total step (for reproducablity)
            if evaluate && s.step % eval_every == 0
                @info "evaluating observables"
                polyakov_loop_correlator = compute_polykov_correlator(s)
                # save computed progress to disk
                write(f, "step$(s.step)", polyakov_loop_correlator)
            end
        end
    end
    @info "done running simulation"
    # saving the final simulation state for restarts
    filename_simulation = "simulation.jld"
    @info "saving simulation state to $filename_simulation"
    save_object(filename_simulation, s)
end

end


function test()
    s = GluoDynamics.GluoDynamicsLattice(8_5_1996, 10, 10, 1000, 1.0, 100, 100, 0.01)
    GluoDynamics.main_loop!(s, 200, true, 100)
end
