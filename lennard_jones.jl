module LennardJones

using StaticArrays
using PyPlot
using LinearAlgebra
using Raylib.Binding
using Raylib

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
    const sigma :: Float64
    const epsilon :: Float64
    const dt :: Float64
    const maxwell_std :: Float64
    const box_size :: Float64
    const thermostat_every_nth_iteration :: Int
    const m :: Float64
    step :: Int
end

function add_particle_to_box!(index::Int, particle::Particle, simulation::Simulation)
    i = CartesianIndex(floor.(Int, particle.x ./ simulation.box_size) .+ 1...)
    push!(simulation.boxes[i].particle_indicies, index)
    particle.box_index = i
end

function foreach_particle_pair(f::Function, simulation::Simulation)
    already_conputed = falses(length(simulation.particles), length(simulation.particles))
    for particle_index in 1:length(simulation.particles)
        p = simulation.particles[particle_index]
        for i in p.box_index[1]-1 : p.box_index[1]+1
            for j in p.box_index[2]-1 : p.box_index[2]+1
                for k in p.box_index[3]-1 : p.box_index[3]+1
                    if i < 1 || i > size(simulation.boxes, 1) || j < 1 || j > size(simulation.boxes, 2) || k < 1 || k > size(simulation.boxes, 3)
                        continue
                    end
                    for other_particle_index in 1:length(simulation.boxes[i, j, k].particle_indicies)
                        if !already_conputed[particle_index, other_particle_index] && particle_index != other_particle_index
                            already_conputed[other_particle_index, particle_index] = true
                            q = simulation.particles[other_particle_index]
                            f(p, q)
                        end
                    end
                end
            end
        end
    end
end

function compute_forces!(simulation::Simulation)
    F = zero(MVector{3, Float64})
    for p in simulation.particles
        p.F = zero(MVector{3, Float64})
    end
    foreach_particle_pair(simulation) do p, q
        diff = p.x - q.x
        dist = norm(diff)
        d = dist / simulation.sigma
        V_diff = 5 * simulation.epsilon * (12 * d^11 - 6 * d^5)
        F = diff / dist * V_diff
        p.F += F
        p.F -= F
    end
end

function maxwell(simulation::Simulation)
    return MVector(randn(), randn(), randn()) * simulation.maxwell_std
end

function make_step!(simulation::Simulation)
    for p in simulation.particles
        p.x += simulation.dt * p.v + 0.5 * simulation.dt^2 * p.F / simulation.m
    end

    for p in simulation.particles
        p.F_old = p.F
    end

    compute_forces!(simulation)

    for p in simulation.particles
        p.v += simulation.dt * (p.F + p.F_old)
    end

    for p in simulation.particles
        for d in 1:3
            if p.x[d] <= 0.0
                p.x[d] = 0.0 - p.x[d]
                p.v *= -1
                if p.x[d] >= simulation.L
                    p.x[d] = simulation.sigma
                    @warn "particle too fast"
                end
            end
            if p.x[d] >= simulation.L
                p.x[d] = simulation.L - (p.x[d] - simulation.L)
                p.v *= -1
                if p.x[d] <= 0.0
                    p.x[d] = simulation.L - simulation.sigma
                    @warn "particle too fast"
                end
            end
        end
    end

    for box in simulation.boxes
        empty!(box.particle_indicies)
    end

    for (i, particle) in enumerate(simulation.particles)
        add_particle_to_box!(i, particle, simulation)
    end

    if simulation.step % simulation.thermostat_every_nth_iteration == 0
        i = rand(1:length(simulation.particles))
        simulation.particles[i].v = maxwell(simulation)

    end
    simulation.step += 1
end

function init()
    nparticles = 500
    L = 1.0
    T = 10.0
    m = 1.0
    epsilon = 2.0
    sigma = 1e-2
    std_maxwell = sqrt(T / m)

    nboxes = 10
    box_size = L / nboxes
    dt = 1e-4
    thermostat_every_n_iteration = 10

    boxes = [Box(Int[], CartesianIndex(i, j, k))
             for i = 1:nboxes, j = 1:nboxes, k = 1:nboxes]
    simulation = Simulation([], boxes, L, sigma, epsilon, dt,
                            std_maxwell, box_size, thermostat_every_n_iteration, m, 0)

    for _ in 1:nparticles
        x = MVector(rand(), rand(), rand()) * L
        push!(simulation.particles, Particle(x,
                                             maxwell(simulation),
                                             zero(MVector{3, Float64}),
                                             zero(MVector{3, Float64}),
                                             CartesianIndex(-1, -1, -1)))
    end

    for (i ,particle) in enumerate(simulation.particles)
        add_particle_to_box!(i, particle, simulation)
    end

    compute_forces!(simulation)

    return simulation
end

function main()
    simulation = init()

    time_per_frame = 1e-2
    screen_width = 2*800
    screen_height = 2*450
    cam_dist = 2.0
    steps_per_frame = floor(Int, time_per_frame / simulation.dt)

    InitWindow(screen_width, screen_height, "Lennard Jones Fluid")
    position = RayVector3(cam_dist, cam_dist - 0.1, cam_dist + 0.1)
    target = RayVector3(simulation.L/2, simulation.L/2, simulation.L/2)
    up = RayVector3(0.0, 1.0, 0.0)
    fovy = 45.0
    projection = CAMERA_PERSPECTIVE
    camera = RayCamera3D(position, target, up, fovy, projection)

    SetTargetFPS(60)

    while !WindowShouldClose()
        BeginDrawing()
        ClearBackground(Raylib.WHITE)
        BeginMode3D(camera)
        DrawCubeWires(target, simulation.L, simulation.L, simulation.L, Raylib.MAROON)
        for p in simulation.particles
            pos = RayVector3(p.x[1], p.x[2], p.x[3])
            DrawSphere(pos, 5e-3, Raylib.RED)
        end
        EndMode3D()
        EndDrawing()

        println("step = $(simulation.step)")
        for _ in 1:steps_per_frame
            make_step!(simulation)
        end
    end
end


end

LennardJones.main()
