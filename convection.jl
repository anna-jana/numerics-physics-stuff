# Simulation of a temperature gradient driven (natural) convection cell.
# The boussinesq approximation is used for the density and incompressibility is enforced by chorin projection.
# The equations of motion are used in dimensionless form.
# The discretisation is done with a finite difference scheme
# using upwind differencing for the advection terms and implicit treatment of the diffusion terms.
# Linear systems are solved using gauss-seidel method with overrelaxation for the pressure computation.
# The top and bottom boundaries are hard walls with heating from below.
# In the x-direction periodic boundary conditions are used.

module Convection

using StaticArrays
using LinearAlgebra
using Base
using PyPlot

Base.@kwdef mutable struct Sim
    # geometry and boundary conditions
    const Lx :: Float64
    const Ly :: Float64
    const T_top :: Float64
    const T_bottom :: Float64
    # physical (material) properties
    const mean_density :: Float64
    const kinematic_viscosity :: Float64
    const thermal_conductivity :: Float64
    const heat_capacity_V :: Float64
    const thermal_expansion_coeff :: Float64
    const gravity_accel :: Float64
    # dimensionless parameters
    const C_1 :: Float64
    const C_2 :: Float64
    const C_3 :: Float64
    # dimensionless parameter for the convection
    const Ra :: Float64
    const T0 :: Float64
    # discretisation parameter
    const Nx :: Int
    const Ny :: Int
    const dx :: Float64
    const dy :: Float64
    const tspan :: Float64
    const dt :: Float64
    const nsteps :: Int
    const eps :: Float64
    const over_relaxation :: Float64

    # state
    velocity :: Array{SVector{2, Float64}, 2}
    velocity_without_pressure :: Array{SVector{2, Float64}, 2}
    temperature :: Array{Float64, 2}
    div_v :: Array{Float64, 2}
    pressure :: Array{Float64, 2}
    explicit_velocity_rhs :: Array{SVector{2, Float64}, 2}
    explicit_temperature_rhs :: Array{Float64, 2}
    step :: Int
    time :: Float64
end

function init() :: Sim
    println("initializing simulation...")
    # discretisation parameter
    Nx = 100
    Ny = 50
    over_relaxation = 1.5
    eps = 1e-10
    # time in internal units
    dt = 1e-5
    tspan = 1e-1

    # physical parameters (in physical units)
    Lx = 10.0
    Ly = 5.0
    T_top = 1.0
    T_bottom = 10.0
    mean_density = 1.0
    kinematic_viscosity = 10.0
    thermal_conductivity = 10.0
    heat_capacity_V = 1.0
    heat_capacity_P = 1.0
    thermal_expansion_coeff = 10.0
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
    Ra = gravity_accel * thermal_expansion_coeff / (kinematic_viscosity * thermal_conductivity) * Delta_T * Ly^3
    @show Ra, Ra < 1700.0
    @show - (T_top - T_bottom) / Ly < gravity_accel * thermal_expansion_coeff * T0 / heat_capacity_P

    # derived discretisation parameters
    dx = (Lx / Ly) / Nx
    dy = 1.0 / (Ny - 1)
    nsteps = ceil(Int, tspan / dt)

    # allocate arrays for the differnent fields
    velocity = zeros(SVector{2, Float64}, (Nx, Ny))
    velocity_without_pressure = zeros(SVector{2, Float64}, (Nx, Ny))

    temperature = zeros(Nx, Ny) .+ T_top
    temperature[:, Ny] .= T_bottom
    temperature[:, :] ./= Delta_T
    temperature[div(Nx, 4):end-div(Nx, 4), end - 1] .= 2*temperature[1, end]

    div_v = zeros(Nx, Ny - 1)
    pressure = ones(Nx, Ny - 1)

    explicit_velocity_rhs = similar(velocity)
    explicit_temperature_rhs = similar(temperature)

    println(" done")

    return Sim(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, T_top=T_top, T_bottom=T_bottom, dt=dt, tspan=tspan,
        mean_density=mean_density, kinematic_viscosity=kinematic_viscosity,
        thermal_conductivity=thermal_conductivity, heat_capacity_V=heat_capacity_V,
        thermal_expansion_coeff=thermal_expansion_coeff,
        gravity_accel=gravity_accel,
        C_1=C_1, C_2=C_2, C_3=C_3, Ra=Ra, T0=T0,
        dx=dx, dy=dy, nsteps=nsteps,
        eps=eps, over_relaxation=over_relaxation,
        velocity=velocity, velocity_without_pressure=velocity_without_pressure,
        temperature=temperature,
        div_v=div_v, pressure=pressure,
        explicit_velocity_rhs=explicit_velocity_rhs, explicit_temperature_rhs=explicit_temperature_rhs,
        step=0, time=0.0,
    )
end

function upwind_grad_at(sim::Sim, field, i, j)
    v = sim.velocity[i, j]
    if v[1] > 0.0
        @inbounds diff_x = (field[i, j] - field[mod1(i - 1, sim.Nx), j]) / sim.dx
    else
        @inbounds diff_x = (field[mod1(i + 1, sim.Nx), j] - field[i, j]) / sim.dx
    end
    if v[2] > 0.0
        @inbounds diff_y = (field[i, j] - field[i, j - 1]) / sim.dy
    else
        @inbounds diff_y = (field[i, j + 1] - field[i, j]) / sim.dy
    end
    return SVector(diff_x, diff_y)
end

function time_step!(sim::Sim)
    # compute the explicit part
    println("explicit part for velocity and temperature")
    @inbounds for j in 2:sim.Ny-1
        @inbounds @simd for i in 1:sim.Nx
            v = sim.velocity[i, j]
            T = sim.temperature[i, j]
            grad_v = upwind_grad_at(sim, sim.velocity, i, j)
            advection_velocity = SVector(
             v[1] * grad_v[1][1] + v[2] * grad_v[2][1],
             v[1] * grad_v[1][2] + v[2] * grad_v[2][2],
            )
            sim.explicit_velocity_rhs[i, j] = v / sim.dt - advection_velocity
            sim.explicit_velocity_rhs[i, j] += SVector(0.0, - sim.C_2 * (T - sim.T0))
            grad_T = upwind_grad_at(sim, sim.temperature, i, j)
            sim.explicit_temperature_rhs[i, j] = T / sim.dt - dot(v, grad_T) # advection
        end
    end

    # gauss seidel - overrelaxation loop to solve the implicit equation for velocity
    println("implicit velocity")
    prefactor_velocity = 1 / sim.dt + sim.C_1 * 2 * (1 / sim.dx^2 + 1 / sim.dy^2)
    copy!(sim.velocity_without_pressure, sim.velocity) # initial guess is the velocity from the last step
    while true
        mse = 0.0
        mean_speed = 0.0
        @inbounds for j in 2:sim.Ny-1
            @inbounds for i in 1:sim.Nx
                neighbor_sum = (
                    sim.velocity_without_pressure[mod1(i - 1, sim.Nx), j] / sim.dx^2 +
                    sim.velocity_without_pressure[mod1(i + 1, sim.Nx), j] / sim.dx^2 +
                    sim.velocity_without_pressure[i, j - 1] / sim.dy^2 +
                    sim.velocity_without_pressure[i, j + 1] / sim.dy^2
                )
                new_val = sim.explicit_velocity_rhs[i, j] + sim.C_1 * neighbor_sum
                new_val /= prefactor_velocity
                mse += norm(sim.velocity_without_pressure[i, j] - new_val)^2
                mean_speed += norm(new_val)
                sim.velocity_without_pressure[i, j] = new_val
            end
        end
        if isapprox(mean_speed, 0.0)
            mse = sqrt(mse / (sim.Nx * (sim.Ny - 2)))
        else
            mean_speed /= sim.Nx * (sim.Ny - 2)
            mse = sqrt(mse / (sim.Nx * (sim.Ny - 2))) / mean_speed
        end
        if !isfinite(mse)
            error("divergence of implicit velocity solver")
        end
        @show mse
        if mse < sim.eps
            break
        end
    end

    # gauss seidel - overrelaxation loop to solve the implicit equation for temperature
    println("implicit temperature")
    prefactor_temperature = 1 / sim.dt + sim.C_1 * 2 * (1 / sim.dx^2 + 1 / sim.dy^2)
    mean_temperature = 0.0
    while true
        mse = 0.0
        @inbounds for j in 2:sim.Ny-1
            @inbounds for i in 1:sim.Nx
                neighbor_sum = (
                    sim.temperature[mod1(i - 1, sim.Nx), j] / sim.dx^2 +
                    sim.temperature[mod1(i + 1, sim.Nx), j] / sim.dx^2 +
                    sim.temperature[i, j - 1] / sim.dy^2 +
                    sim.temperature[i, j + 1] / sim.dy^2
                )
                new_val = sim.explicit_temperature_rhs[i, j] + sim.C_3 * neighbor_sum
                new_val /= prefactor_temperature
                mse += (sim.temperature[i, j] - new_val)^2
                sim.temperature[i, j] = new_val
                mean_temperature += new_val
            end
        end
        mean_temperature /= sim.Nx * (sim.Ny - 2)
        mse = sqrt(mse / (sim.Nx * (sim.Ny - 2))) / mean_temperature
        if !isfinite(mse)
            error("divergence of implicit temperature solver")
        end
        @show mse
        if mse < sim.eps
            break
        end
    end
end

function chorin_projection!(sim::Sim)
    # compute the divergence of the updated velocity v*
    println("chorin_projection")
    v = sim.velocity_without_pressure
    @inbounds for j in 1:sim.Ny - 1
        @inbounds @simd for i in 1:sim.Nx
            sim.div_v[i, j] = (
               (v[mod1(i + 1, sim.Nx), j][1] - v[i, j][1] +
                v[mod1(i + 1, sim.Nx), j + 1][1] - v[i, j + 1][1]) / (2*sim.dx) +
               (v[i, j + 1][2] - v[i, j][2] +
                v[mod1(i + 1, sim.Nx), j + 1][2] - v[mod1(i + 1, sim.Nx), j][2]) / (2*sim.dy)
            )
        end
    end

    # solve laplace P = div v* using gauss-seidel iteration
    iteration = 0
    P = sim.pressure
    while true
        mse = 0.0
        mean_pressure = 0.0
        # top
        @inbounds for i in 1:sim.Nx
            P_sum = (
                     P[mod1(i - 1, sim.Nx), 1] / sim.dx^2 +
                     P[mod1(i + 1, sim.Nx), 1] / sim.dx^2 +
                     P[i, 2] / sim.dy^2
            )
            new_val = (sim.div_v[i, 1] - P_sum) / (-2/sim.dx^2 - 1/sim.dy^2)
            new_val = (1 - sim.over_relaxation) * P[i, 1] + sim.over_relaxation * new_val
            mse += (P[i, 1] - new_val)^2 / new_val^2
            P[i, 1] = new_val
            mean_pressure += new_val
        end

        # bottom
        @inbounds for i in 1:sim.Nx
            P_sum = (
                     P[mod1(i - 1, sim.Nx), end] / sim.dx^2 +
                     P[mod1(i + 1, sim.Nx), end] / sim.dx^2 +
                     P[i, end - 1] / sim.dy^2
            )
            new_val = (sim.div_v[i, end] - P_sum) / (-2/sim.dx^2 - 1/sim.dy^2)
            new_val = (1 - sim.over_relaxation) * P[i, end] + sim.over_relaxation * new_val
            mse += (P[i, end] - new_val)^2 / new_val^2
            P[i, end] = new_val
            mean_pressure += new_val
        end

        # interior
        @inbounds for j in 2:sim.Ny - 2
            @inbounds for i in 1:sim.Nx
                P_sum = (
                         P[mod1(i - 1, sim.Nx), j] / sim.dx^2 +
                         P[mod1(i + 1, sim.Nx), j] / sim.dx^2 +
                         P[i, j - 1] / sim.dy^2 +
                         P[i, j + 1] / sim.dy^2
                )
                new_val = (sim.div_v[i, j] - P_sum) / (- 2*(1/sim.dx^2 + 1/sim.dy^2))
                new_val = (1 - sim.over_relaxation) * P[i, j] + sim.over_relaxation * new_val
                mse += (P[i, j] - new_val)^2
                P[i, j] = new_val
                mean_pressure += new_val
            end
        end
        mean_pressure /= sim.Nx * (sim.Ny - 1)
        mse = sqrt(mse / (sim.Nx * (sim.Ny - 1))) / mean_pressure
        print("\rmse = $mse, iteration = $iteration")
        if !isfinite(mse)
            error("divergence of chorin_projection solver")
        end
        if mse <= sim.eps
            break
        end
        iteration += 1
    end
    println("")

    # correct velocity by v = v* - grad P
    @inbounds for j in 2:sim.Ny - 1
        @inbounds @simd for i in 1:sim.Nx
            diff_x = (P[i, j - 1] - P[mod1(i - 1, sim.Nx), j - 1] + P[i, j] - P[mod1(i - 1, sim.Nx), j]) / (2*sim.dx)
            diff_y = (P[mod1(i - 1, sim.Nx), j] - P[mod1(i - 1, sim.Nx), j - 1] + P[i, j] - P[i, j - 1]) / (2*sim.dy)
            grad_P = SVector(diff_x, diff_y)
            # grad_P += SVector(0.0, sim.mean_density * sim.gravity_accel)
            # sim.velocity[i, j] = sim.velocity_without_pressure[i, j] - sim.dt * grad_P
            sim.velocity[i, j] = sim.velocity_without_pressure[i, j] - grad_P
        end
    end
end

function step!(sim::Sim)
    println("step = $(sim.step) / $(sim.nsteps)")
    time_step!(sim)
    chorin_projection!(sim)
    sim.step += 1
    sim.time += sim.dt
end

function plot(sim::Sim)
    xs = range(0, sim.Lx, sim.Nx)
    ys = range(0, sim.Ly, sim.Ny)

    figure()
    contourf(xs, ys, transpose(sim.temperature))
    colorbar(orientation="horizontal")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    gca().set_aspect("equal")
    title("temperature")
    tight_layout()

    figure()
    contourf(xs, ys, transpose([norm(v) for v in sim.velocity]))
    quiver(xs, ys, transpose([v[1] for v in sim.velocity]),
           transpose([-v[2] for v in sim.velocity]), color="red")
    gca().invert_yaxis()
    gca().set_aspect("equal")
    colorbar(label="|v|", orientation="horizontal")
    xlabel("x")
    ylabel("y")
    title("velocity")
    tight_layout()

    show()
end

end

@time begin
    print("starting simulation...")
    sim = Convection.init()
    for i in 1:sim.nsteps
        Convection.step!(sim)
    end
    print("ending simulation.")
end
println("plotting...")
Convection.plot(sim)
