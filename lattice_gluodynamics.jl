module GluoDynamics

using Random
using StaticArrays
using PyPlot
using LinearAlgebra
using JLD2
using FileIO

include("su3.jl")
include("timeseries.jl")

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
    # indicies for evlaulation of the plaquett observable
    plaquett_indicies::Array{CartesianIndex{5},7}
    # state for the evaluation of polyakov loops
    polyakov::Array{ComplexF64,3}
    polyakov_corr::Vector{Vector{Float64}}
    # polyakov_corr_single::Vector{Float64}
    counts::Vector{Int}
end

############################## links ############################
# index into the link variables (gluon field)
function add_offset!(coord::Vector{Int}, signed_direction::Int, dims)
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
    @info "generating stable indicies"
    dims = (N, N, N, Nt)
    stable_indicies = Array{CartesianIndex{5}}(undef, 6, 3, 4, dims...)
    Threads.@threads for n in CartesianIndices(dims) # each lattice location
        for mu in 1:4 # link direction
            nu_i = 1
            for nu in 1:4 # other direction of the placett
                if mu != nu
                    stable_indicies[1, nu_i, mu, n] = get_link_index(n, nu, [mu], dims)
                    stable_indicies[2, nu_i, mu, n] = get_link_index(n, -mu, [nu, mu], dims)
                    stable_indicies[3, nu_i, mu, n] = get_link_index(n, -nu, [nu], dims)
                    stable_indicies[4, nu_i, mu, n] = get_link_index(n, -nu, [mu], dims)
                    stable_indicies[5, nu_i, mu, n] = get_link_index(n, -mu, [-nu, mu], dims)
                    stable_indicies[6, nu_i, mu, n] = get_link_index(n, nu, [-nu], dims)
                    nu_i += 1
                end
            end
        end
    end
    return stable_indicies
end

# compute the action difference when changing one link
@inline function compute_stable(s::GluoDynamicsLattice, i::CartesianIndex{5})
    A = zero(SMatrix{3,3,ComplexF64})
    @inbounds for nu_i in 1:3
        A += to_matrix(s.U[s.stable_indicies[1, nu_i, i]] *
                       s.U[s.stable_indicies[2, nu_i, i]] *
                       s.U[s.stable_indicies[3, nu_i, i]])
        A += to_matrix(s.U[s.stable_indicies[4, nu_i, i]] *
                       s.U[s.stable_indicies[5, nu_i, i]] *
                       s.U[s.stable_indicies[6, nu_i, i]])
    end
    return A
end

@inline real_of_mul(a::ComplexF64, b::ComplexF64)::Float64 = a.re * b.re - a.im * b.im

@inline function compute_action_diff(s::GluoDynamicsLattice, i::CartesianIndex{5}, U_new::SU3)
    A = compute_stable(s, i)
    @inbounds U = to_matrix(s.U[i])
    U_new_matrix = to_matrix(U_new)
    # return -s.beta / s.N * real(tr((U_new_matrix - U) * A))
    U_diff = U_new_matrix - U
    @inbounds trace = (
        real_of_mul(U_diff[1, 1], A[1, 1]) + real_of_mul(U_diff[1, 2], A[2, 1]) + real_of_mul(U_diff[1, 3], A[3, 1]) +
        real_of_mul(U_diff[2, 1], A[1, 2]) + real_of_mul(U_diff[2, 2], A[2, 2]) + real_of_mul(U_diff[2, 3], A[3, 2]) +
        real_of_mul(U_diff[3, 1], A[1, 3]) + real_of_mul(U_diff[3, 2], A[2, 3]) + real_of_mul(U_diff[3, 3], A[3, 3])
    )
    return -s.beta / s.N * trace
end

######################## observables ###########################
function generate_plaquett_indicies(N, Nt)
    @info "generating plaquett indicies"
    dims = (N, N, N, Nt)
    plaquett_indicies = Array{CartesianIndex{5}}(undef, 4, 3, 4, dims...)
    Threads.@threads for n in CartesianIndices(dims)
        @inbounds for mu in 1:4
            nu_i = 1
            for nu in 1:4
                if nu != mu
                    plaquett_indicies[1, nu_i, mu, n] = get_link_index(n, mu, [], dims)
                    plaquett_indicies[2, nu_i, mu, n] = get_link_index(n, nu, [mu], dims)
                    plaquett_indicies[3, nu_i, mu, n] = get_link_index(n, mu, [nu], dims)
                    plaquett_indicies[4, nu_i, mu, n] = get_link_index(n, nu, [], dims)
                    nu_i += 1
                end
            end
        end
    end
    return plaquett_indicies
end

function eval_plaquetts(s::GluoDynamicsLattice)
    @info "evaluating plaquetts"
    P_atomic = Threads.Atomic{Float64}(0.0)
    Threads.@threads for i in CartesianIndices(s.plaquett_indicies[1, :, :, :, :, :, :])
        P_local = real(tr(
            s.U[s.plaquett_indicies[1, i]] *
            s.U[s.plaquett_indicies[2, i]] *
            s.U[s.plaquett_indicies[3, i]]' *
            s.U[s.plaquett_indicies[4, i]]'
        ))
        Threads.atomic_add!(P_atomic, P_local)
    end
    P = P_atomic[]
    return P / (6 * s.N^3 * s.Nt)
end

@inline eval_polyakov_loop(s, n) = tr(prod(@inbounds s.U[n, nt, 4] for nt in 1:s.Nt))

@inline cyclic_dist_squared_1d(x1, x2, N) = min((x1 - x2)^2, (N - x1 + x2)^2, (N - x2 + x1)^2)

@inline function cyclic_dist_squared(n, m, N)
    @inbounds return (
        cyclic_dist_squared_1d(n[1], m[1], N) +
        cyclic_dist_squared_1d(n[2], m[2], N) +
        cyclic_dist_squared_1d(n[1], m[2], N)
    )
end

function update_polyakov_correlator!(s::GluoDynamicsLattice)
    @info "evaluating polyako loops"
    # eval each polyakov loop
    Threads.@threads for n in CartesianIndices((s.N, s.N, s.N))
        s.polyakov[n] = eval_polyakov_loop(s, n)
    end
    # compute their correlator as a histogram
    # only the distance between the points matters bc of translation invariance
    fill!(s.counts, 0)
    push!(s.polyakov_corr, zeros(length(s.counts)))
    @inbounds for n in CartesianIndices(s.polyakov)
        for m in CartesianIndices(s.polyakov)
            d2 = cyclic_dist_squared(n, m, s.N)
            val = s.polyakov[n] * s.polyakov[m]'
            # if !isapprox(val, 0.0, atol=1e-10)
            #     @warn "imaginary part of polyakov loop correlator is $(imag(val))"
            # end
            s.polyakov_corr[end][d2] += real(val)
            s.counts[d2] += 1
        end
    end
    for i in eachindex(s.counts)
        if s.counts[i] != 0
            s.polyakov_corr[end][i] /= s.counts[i]
        end
    end
end

############################ initialization ########################
# cold start = low temp = 1 on each link
cold_start(N, Nt) = ones(SU3, (N, N, N, Nt, 4))

# generate mc steps
function generate_possible_mc_steps(rng, n, epsilon)::Vector{SU3}
    choices_half = [random_su3_close_to_1(rng, epsilon) for _ in 1:n]
    inv_half = inv.(choices_half)
    return vcat(choices_half, inv_half)
end

function GluoDynamicsLattice(seed, N, Nt, nchoices, beta, reproject_every, new_choices_every, epsilon)
    @info "building new lattice simulation $N^3*$Nt with seed $seed @ beta = $beta"
    rng = Xoshiro(seed)
    polyakov_corr_size = cld(3 * (N - 1)^2, 2)
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
        plaquett_indicies=generate_plaquett_indicies(N, Nt),
        polyakov=Array{Float64}(undef, N, N, N),
        polyakov_corr=Vector{Float64}[],
        counts=zeros(Int, polyakov_corr_size),
    )
end

######################### mc algorithm ########################
@inline function mc_step_metropolis!(s::GluoDynamicsLattice, i::CartesianIndex{5})
    # metropolis for a first test
    # use a random step
    X = rand(s.choices)
    U = s.U[i]
    U_new = X * U
    # compute action difference for acceptance propability
    Delta_S = compute_action_diff(s, i, U_new)
    # accept
    p = rand()
    if p <= exp(-Delta_S)
        s.U[i] = U_new
    end
end

function mc_sweep_metropolis!(s::GluoDynamicsLattice)
    for n in CartesianIndices(s.U)
        mc_step_metropolis!(s, n)
    end
    s.step += 1
end

##################### the main mc loop #########################
const info_every_secs = 1

function advance!(s::GluoDynamicsLattice, nsteps)
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
        mc_sweep_metropolis!(s)
    end
end

function run_simple_simulation(beta;
    equilibrating_steps=50, discarded_updates=1, nsamples=200,
    filename_polyakov="polyakov_loops$beta.jld",
    filename_plaquetts="plaquetts$beta.jld",
    filename_simulation="simulation$beta.jld",
)
    s = GluoDynamicsLattice(8_5_1996, 12, 12, 1000, beta, 100, 100, 0.01)

    @info "running simulation"

    @info "equilibrating for $equilibrating_steps steps"
    advance!(s, equilibrating_steps)

    @info "collecting data"
    plaquetts = Float64[]
    for nth_sample in 1:nsamples
        # discard step inbetween observable evaluations
        @info "collecting sample $nth_sample / $nsamples"
        advance!(s, discarded_updates + 1)

        # observables
        @info "evaluating observables"

        P = eval_plaquetts(s)
        push!(plaquetts, P)

        update_polyakov_correlator!(s)
    end

    @info "done running simulation"

    @info "saving dataseries of plaquetts in $filename_plaquetts"
    save_object(filename_plaquetts, plaquetts)

    @info "saving dataseries of polyakov loops in $filename_polyakov"
    s.polyakov_corr ./= nsamples
    save_object(filename_polyakov, s.polyakov_corr)

    # saving the final simulation state for restarts
    @info "saving simulation state to $filename_simulation"
    save_object(filename_simulation, s)
end

end # module GluoDynamics

using JLD2
using PyPlot

function analysis()
    beta = 1.0
    Nt = 12
    P = load_object("polyakov_loops$beta.jld")
    d2 = collect(1:length(P[1]))
    for p in P
        plot(log.(abs.(p[p.!=0.0])), ".k")
    end
end
