module Argon

using StaticArrays
using LinearAlgebra
using Random

# unit of mass such that the mass of a particle is 1
# the unit of length is sigma
# hence the size of each box is 1
# the unit of time is (m * sigma^2 / epsilon)**0.5

mutable struct Particle
    x :: MVector{3, Float64}
    v :: MVector{3, Float64}
    F :: MVector{3, Float64}
    F_old :: MVector{3, Float64}
    box_index :: CartesianIndex{3}
end

struct Box
    particle_indicies :: Vector{Int}
    box_index :: CartesianIndex{3}
end

mutable struct Simulation
    particles :: Vector{Particle}
    boxes :: Array{Box, 3}
    const L :: Float64
    const dt :: Float64
    const T :: Float64
    step :: Int
end

function add_particle_to_box!(index::Int, particle::Particle, simulation::Simulation)
    i = CartesianIndex(floor.(Int, particle.x) .+ 1...)
    push!(simulation.boxes[i].particle_indicies, index)
    particle.box_index = i
end

function foreach_particle_pair(f::Function, simulation::Simulation)
    for (i, p) in enumerate(simulation.particles)
        for (j, q) in enumerate(simulation.particles)
            if i != j
                diff = p.x - q.x
                # minimum image convetion
                for i in 1:3
                    if abs(diff[i]) > simulation.L / 2
                        diff[i] -= round(Int, diff[i] / simulation.L) * simulation.L
                    end
                end
                d = norm(diff)
                f(p, q, diff, d)
            end
        end
    end

    return

    already_conputed = falses(length(simulation.particles), length(simulation.particles))

    for (particle_index, p) in enumerate(simulation.particles)
        @inbounds for i in p.box_index[1]-1 : p.box_index[1]+1
            for j in p.box_index[2]-1 : p.box_index[2]+1
                for k in p.box_index[3]-1 : p.box_index[3]+1
                    i = mod1(i, size(simulation.boxes, 1))
                    j = mod1(j, size(simulation.boxes, 2))
                    k = mod1(k, size(simulation.boxes, 3))

                    for other_particle_index in 1:length(simulation.boxes[i, j, k].particle_indicies)
                        if !already_conputed[particle_index, other_particle_index] && particle_index != other_particle_index
                            already_conputed[other_particle_index, particle_index] = true
                            q = simulation.particles[other_particle_index]
                            diff = p.x - q.x
                            # minimum image convetion
                            for i in 1:3
                                if abs(diff[i]) > simulation.L / 2
                                    diff[i] -= round(Int, diff[i] / simulation.L) * simulation.L
                                end
                            end
                            d = norm(diff)
                            f(p, q, diff, d)
                        end
                    end
                end
            end
        end
    end
end

function compute_forces!(simulation::Simulation)
    for p in simulation.particles
        p.F = zero(MVector{3, Float64})
    end
    V = 0.0
    foreach_particle_pair(simulation) do p, q, diff, d
        F_norm2 = 0.0
        @inbounds for i in 1:3
            F = - diff[i] * (48 * d^(-14) - 24 * d^(-8))
            p.F[i] -= F
            q.F[i] += F
            F_norm2 += F^2
        end
        V += d * sqrt(F_norm2)
    end
    return V
end

function make_step!(simulation::Simulation)
    for p in simulation.particles
        p.x += simulation.dt * p.v + 0.5 * simulation.dt^2 * p.F
        p.x = mod.(p.x, simulation.L)
    end

    for p in simulation.particles
        p.F_old = p.F
    end

    V = compute_forces!(simulation)

    for p in simulation.particles
        p.v += simulation.dt * (p.F + p.F_old) * 0.5
    end

    for box in simulation.boxes
        empty!(box.particle_indicies)
    end

    for (i, particle) in enumerate(simulation.particles)
        add_particle_to_box!(i, particle, simulation)
    end

    simulation.step += 1

    return V
end

function rescale_velocity_to_temperature!(simulation::Simulation)
    E_kin = sum(p -> norm(p.v)^2, simulation.particles)
    @show E_kin
    N = length(simulation.particles)
    lambda = sqrt((N - 1) * 3 * simulation.T / E_kin)
    for p in simulation.particles
        p.v .*= lambda
    end
end

function run(rho, T, M)
    Random.seed!(102133)
    N = 4 * M^3
    # rho = L^3 / (4*M^3) = 1/4 * (L/M)^3
    # L= M * (4rho)^(1/3)
    L = M * cbrt(4*rho)

    dt = 0.004 / 2
    init_time = 10.0
    rescale_every_n_steps = 10
    production_time = 20.0

    nboxes = ceil(Int, L)
    boxes = [Box(Int[], i) for i in CartesianIndices((nboxes, nboxes, nboxes))]
    simulation = Simulation([], boxes, L, dt, T, 0)

    function add_particle_at(x)
        v = MVector(randn(), randn(), randn()) * sqrt(T) # maxwell distribution
        push!(simulation.particles, Particle(x, v, zero(MVector{3, Float64}), zero(MVector{3, Float64}), CartesianIndex(-1, -1, -1)))
    end

    dx = L / M # periodic bc
    for i in 0:M-1, j in 0:M-1, k in 0:M-1
        add_particle_at(MVector(i,       j,       k)       .* dx)
        add_particle_at(MVector(i + 0.5, j + 0.5, k)       .* dx)
        add_particle_at(MVector(i + 0.5, j,       k + 0.5) .* dx)
        add_particle_at(MVector(i,       j + 0.5, k + 0.5) .* dx)
    end

    # enforce vanishing total velocity/momentum
    total_veloctiy = sum(p -> p.v, simulation.particles)
    total_velocity_per_particle = total_veloctiy / length(simulation.particles)
    for p in simulation.particles
        p.v -= total_velocity_per_particle
    end

    for (i ,particle) in enumerate(simulation.particles)
        add_particle_to_box!(i, particle, simulation)
    end

    compute_forces!(simulation)

    init_steps = ceil(Int, init_time / dt)
    for i in 0:init_steps-1
        println("init step $(i + 1) / $init_steps")
        if i % rescale_every_n_steps == 0
            rescale_velocity_to_temperature!(simulation)
        end
        make_step!(simulation)
    end

    production_steps = ceil(Int, production_time / dt)
    E_kins = Array{Float64}(undef, production_steps)
    E_pots = zeros(production_steps)
    virials = Array{Float64}(undef, production_steps)
    for i in 1:production_steps
        println("production step $i / $production_steps")
        virials[i] = make_step!(simulation)
        E_kins[i] = sum(p -> 0.5 * norm(p.v)^2, simulation.particles)
        foreach_particle_pair(simulation) do _, _, _, d
            E_pots[i] += 4 * (d^(-12) - d^(-6))
        end
    end

    return E_kins, E_pots, virials
end

end

using PyPlot
using Statistics

M = 3
rho = 0.8
E_kins, E_pots, virials = Argon.run(rho, 1.0, M)
N = 4*M^3

T = mean(E_kins) / ((N - 1)*3/2)
E_pot = mean(E_pots / N)

@show rho, T, E_pot
