module Convection

using PyPlot
using StaticArrays
using LinearAlgebra
using Base
using Random

mutable struct Sim
    const Nx :: Int
    const Ny :: Int
    const Lx :: Float64
    const Ly :: Float64
    const T_top :: Float64
    const T_bottom :: Float64
    const dt :: Float64
    const tspan :: Float64
    const mean_density :: Float64
    const kinematic_viscosity :: Float64
    const thermal_conductivity :: Float64
    const heat_capacity_V :: Float64
    const thermal_expansion_coeff :: Float64
    const gravity_accel :: Float64

    const C_1 :: Float64
    const C_2 :: Float64
    const C_3 :: Float64

    const T0 :: Float64

    const dx :: Float64
    const dy :: Float64
    const nsteps :: Int

    velocity :: Array{SVector{2, Float64}, 2}
    velocity_without_pressure :: Array{SVector{2, Float64}, 2}
    velocity_new :: Array{SVector{2, Float64}, 2}
    temperature :: Array{Float64, 2}
    temperature_new :: Array{Float64, 2}

    div_v :: Array{Float64, 2}
    pressure :: Array{Float64, 2}

    velocity_rhs :: Array{SVector{2, Float64}, 2}
    prev_velocity_rhs :: Array{SVector{2, Float64}, 2}
    temperature_rhs :: Array{Float64, 2}
    prev_temperature_rhs :: Array{Float64, 2}

    step :: Int
    time :: Float64
end

# TODO: move constant paramters to arguments of init
# TODO: make this a constructor of Sim
function init() :: Sim
    println("initializing simulation...")
    # parameter
    Nx = 50
    Ny = 25
    # in physical units
    Lx = 2.0
    Ly = 1.0
    T_top = 1.0
    T_bottom = 2.0
    # time in internal units
    dt = 1e-7
    tspan = 0.1
    # in physical units:
    mean_density = 1.0
    kinematic_viscosity = 0.1
    thermal_conductivity = 1.0
    heat_capacity_V = 1.0
    thermal_expansion_coeff = 1.0
    gravity_accel = 10.0

    # computation of characterisitic dimensionless parameters
    thermal_time_scale = Ly / thermal_conductivity
    gravitational_time_scale = sqrt(Ly / gravity_accel)
    time_scale = min(thermal_time_scale, gravitational_time_scale)
    Delta_T = T_bottom - T_top
    C_1 = kinematic_viscosity * time_scale / Ly^2 # this is 1/Re
    C_2 = time_scale^2 / Ly * gravity_accel * thermal_expansion_coeff * Delta_T
    C_3 = thermal_conductivity * time_scale / (mean_density * heat_capacity_V * Ly^2)

    T0 = T_top / Delta_T

    # derived discretisation parameters
    dx = (Lx / Ly) / Nx
    dy = 1.0 / (Ny - 1)
    nsteps = ceil(Int, tspan / dt)

    # allocate arrays for the differnent fields
    velocity = Array{SVector{2, Float64}, 2}(undef, (Nx, Ny))
    for i in eachindex(velocity)
        velocity[i] = zero(eltype(velocity))
    end
    #for j in 2:Ny-1
    #    for i in 1:Nx
    #        velocity[i, j] += SVector((2*rand() - 1), (2*rand() - 1)) / 100.0
    #    end
    #end
    velocity_without_pressure = similar(velocity)
    velocity_new = similar(velocity)
    for i in 1:Nx
        velocity_without_pressure[i, 1] = velocity_new[i, 1] = velocity[i, 1]
        velocity_without_pressure[i, end] = velocity_new[i, end] = velocity[i, end]
    end

    temperature = zeros(Nx, Ny) .+ T_top
    temperature[:, Ny] .= T_bottom
    temperature[:, :] ./= Delta_T
    temperature_new = similar(temperature)
    temperature_new[:, 1] = temperature[:, 1]
    temperature_new[:, end] = temperature[:, end]
    #temperature[:, 2:end-1] .+= rand(Nx, Ny-2) / 100

    div_v = zeros(Nx, Ny)
    #pressure = ones(Nx, Ny)
    pressure = ones(Nx, Ny) .+ 1e-5 .*(2 .* rand(Nx, Ny) .- 1)

    velocity_rhs = Array{SVector{2, Float64}, 2}(undef, (Nx, Ny))
    prev_velocity_rhs = similar(velocity_rhs)
    temperature_rhs = Array{Float64, 2}(undef, (Nx, Ny))
    prev_temperature_rhs = similar(temperature_rhs)

    println("checking CFL condition...")
    min_dt_temperature = 0.5 * dy^2 * C_3
    min_dt_viscosity = 0.5 * dy^2 * C_1
    @show min_dt_temperature
    @show min_dt_viscosity
    @show dt
    @show dt < min_dt_temperature
    @show dt < min_dt_viscosity

    return Sim(
        Nx, Ny, Lx, Ly, T_top, T_bottom, dt, tspan, mean_density, kinematic_viscosity,
        thermal_conductivity, heat_capacity_V, thermal_expansion_coeff,
        gravity_accel,
        C_1, C_2, C_3, T0,
        dx, dy, nsteps,
        velocity, velocity_without_pressure, velocity_new,
        temperature, temperature_new,
        div_v, pressure,
        velocity_rhs, prev_velocity_rhs, temperature_rhs, prev_temperature_rhs,
        0, 0.0,
    )
end

# functions for taking finite differences
function laplace_at(sim:: Sim, field, i, j)
    d2_field_dx2 = (field[mod1(i + 1, sim.Nx), j] - 2 * field[i, j] + field[mod1(i - 1, sim.Nx), j]) / sim.dx^2
    d2_field_dy2 = (field[i, j + 1]               - 2 * field[i, j] + field[i, j - 1])               / sim.dy^2
    return d2_field_dx2 + d2_field_dy2
end

function grad_at(sim::Sim, field, i, j)
    diff_x = (field[mod1(i + 1, sim.Nx), j] - field[mod1(i - 1, sim.Nx), j]) / (2*sim.dx)
    diff_y = (field[i, j + 1]               - field[i, j - 1])               / (2*sim.dy)
    return SVector(diff_x, diff_y)
end

function upwind_grad_at(sim::Sim, field, i, j)
    v = sim.velocity[i, j]
    if v[1] > 0.0
        diff_x = (field[i, j] - field[mod1(i - 1, sim.Nx), j]) / sim.dx
    else
        diff_x = (field[mod1(i + 1, sim.Nx), j] - field[i, j]) / sim.dx
    end
    if v[2] > 0.0
        diff_y = (field[i, j] - field[i, j - 1]) / sim.dy
    else
        diff_y = (field[i, j + 1] - field[i, j]) / sim.dy
    end
    return SVector(diff_x, diff_y)
end

function compute_rhs!(sim::Sim)
    # updating
    for j in 2:sim.Ny-1
        for i in 1:sim.Nx
            v = sim.velocity[i, j]
            T = sim.temperature[i, j]

            # velocity (Navier-Stokes) rhs
            # grad_v = grad_at(sim, sim.velocity, i, j) # ftcs
            grad_v = upwind_grad_at(sim, sim.velocity, i, j)

            advection_velocity = SVector(
             v[1] * grad_v[1][1] + v[2] * grad_v[2][1],
             v[1] * grad_v[1][2] + v[2] * grad_v[2][2],
            )

            sim.velocity_rhs[i, j] = (
                              - advection_velocity
                                # firction/diffusion
                              + sim.C_1 * laplace_at(sim, sim.velocity, i, j)
            )
            # gravity (only y direction)
            sim.velocity_rhs[i, j] += SVector(0.0, - sim.C_2 * (T - sim.T0))

            # temperature advection + diffusion
            sim.temperature_rhs[i, j] = (
                              - dot(v, upwind_grad_at(sim, sim.temperature, i, j)) # advection
                              + sim.C_3 * laplace_at(sim, sim.temperature, i, j) # diffusion
            )

        end
    end
end

function time_integrator_step!(sim)
    for j in 2:sim.Ny - 1
        for i in 1:sim.Nx
            if sim.step == 0
                # explict euler
                sim.velocity_without_pressure[i, j] = sim.velocity[i, j] + sim.dt * sim.velocity_rhs[i, j]
                sim.temperature_new[i, j] = sim.temperature[i, j] + sim.dt * sim.temperature_rhs[i, j]
            else
                # adams bashforth 2nd order
                sim.velocity_without_pressure[i, j] = sim.velocity[i, j] + sim.dt * (3/2 * sim.velocity_rhs[i, j] - 1/2 * sim.prev_velocity_rhs[i, j])
                sim.temperature_new[i, j] = sim.temperature[i, j] + sim.dt * (3/2 * sim.temperature_rhs[i, j] - 1/2 * sim.prev_temperature_rhs[i, j])
            end
        end
    end
    (sim.velocity_rhs, sim.prev_velocity_rhs, sim.temperature_rhs, sim.prev_temperature_rhs) =
        (sim.prev_velocity_rhs, sim.velocity_rhs, sim.prev_temperature_rhs, sim.temperature_rhs)
end

function chorin_projection!(sim::Sim)
    # compute the divergence of the updated velocity v*
    # interior
    for j in 2:sim.Ny - 1
        for i in 1:sim.Nx
            if sim.velocity_without_pressure[i, j][1] > 0
                d_v_x_dx = (sim.velocity_without_pressure[i, j][1] -
                            sim.velocity_without_pressure[mod1(i - 1, sim.Nx), j][1]) / sim.dx
            else
                d_v_x_dx = (sim.velocity_without_pressure[mod1(i + 1, sim.Nx), j][1] -
                            sim.velocity_without_pressure[i, j][1]) / sim.dx
            end

            if sim.velocity_without_pressure[i, j][2] > 0
                d_v_y_dy = (sim.velocity_without_pressure[i, j][2] -
                            sim.velocity_without_pressure[i, j - 1][2]) / sim.dy
            else
                d_v_y_dy = (sim.velocity_without_pressure[i, j][2] -
                            sim.velocity_without_pressure[i, j - 1][2]) / sim.dy
            end

            sim.div_v[i, j] = d_v_x_dx + d_v_y_dy
        end
    end

    sim.div_v[:, :] ./= sim.dt

    # solve laplace P = density / dt * div v* using gauss-seidel iteration
    eps = 1e-5
    omega = 0.8


    while true
        mse = 0.0
        # interior
        for j in 2:sim.Ny - 1
            for i in 1:sim.Nx
                P_sum = (
                         sim.pressure[mod1(i - 1, sim.Nx), j] / sim.dx^2 +
                         sim.pressure[mod1(i + 1, sim.Nx), j] / sim.dx^2 +
                         sim.pressure[i, j - 1] / sim.dy^2 +
                         sim.pressure[i, j + 1] / sim.dy^2
                )
                P_new = (sim.div_v[i, j] - P_sum) / (- 2*(1/sim.dx^2 + 1/sim.dy^2))
                P_new = (1 - omega) * sim.pressure[i, j] + omega * P_new
                # error calculation
                mse += (sim.pressure[i, j] - P_new)^2
                sim.pressure[i, j] = P_new
            end
        end
        # error calculation
        mse /= (sim.Nx * (sim.Ny - 2))
        mse = sqrt(mse)

        clf()
        subplot(2,2,1)
        pcolormesh(sim.pressure)
        title("P")
        colorbar()
        subplot(2,2,2)
        pcolormesh(sim.div_v)
        title("div v")
        colorbar()
        subplot(2,2,3)
        pcolormesh([v[1] for v in sim.velocity])
        title("v_x")
        colorbar()
        subplot(2,2,4)
        pcolormesh([v[2] for v in sim.velocity])
        title("v_y")
        colorbar()
        pause(0.001)

        # convergence
        if mse <= eps
            break
        end
        @show mse, maximum(sim.pressure)
    end
    # top
    for i in 1:sim.Nx
        sim.pressure[i, 1] = sim.pressure[i, 2]
    end
    # bottom
    for i in 1:sim.Nx
        sim.pressure[i, end] = sim.pressure[i, end-1]
    end

    # correct velocity by v = v* + dt / density * grad P
    for j in 2:sim.Ny - 1
        for i in 1:sim.Nx
            grad_P = grad_at(sim, sim.pressure, i, j)
            grad_P += SVector(0.0, sim.mean_density * sim.gravity_accel)
            sim.velocity_new[i, j] = sim.velocity_without_pressure[i, j] - sim.dt * grad_P
        end
    end
end


# TODO: implmenet different schemes:
# ftcs (done)
# upwind differencing (done),
# adams-bashforth timestepping (done),
# Lax Method,
# implicit methods
function step!(sim::Sim)
    println("step = $(sim.step) / $(sim.nsteps)")

    compute_rhs!(sim)
    time_integrator_step!(sim)
    chorin_projection!(sim)

    (sim.velocity_new, sim.temperature_new, sim.velocity, sim.temperature) = (
             sim.velocity, sim.temperature, sim.velocity_new, sim.temperature_new)
    sim.step += 1
    sim.time += sim.dt
end

function plot(sim::Sim)
    println("plotting...")
    xs = range(0, sim.Lx, sim.Nx)
    ys = range(0, sim.Ly, sim.Ny)
    suptitle("thermal convection with boussinesq approximation and chorin projection")

    subplot(3,1,1)
    pcolormesh(xs, ys, transpose(sim.temperature))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    title("temperature")

    subplot(3,1,2)
    pcolormesh(xs, ys, transpose([v[1] for v in sim.velocity]))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    title("velocity x")

    subplot(3,1,3)
    pcolormesh(xs, ys, transpose([v[2] for v in sim.velocity]))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    title("velocity y")

    tight_layout()
end

end

using PyPlot
figure()
sim = Convection.init()
for i in 1:sim.nsteps
    Convection.step!(sim)
    if false # i % 100 == 0
        clf()
        Convection.plot(sim)
        pause(0.0001)
    end
end
