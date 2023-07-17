module Convection

using PyPlot
using StaticArrays
using LinearAlgebra

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
    const Ra :: Float64
    const T0 :: Float64
    const dx :: Float64
    const dy :: Float64
    const nsteps :: Int

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

# TODO: move constant paramters to arguments of init
# TODO: make this a constructor of Sim
function init() :: Sim
    println("initializing simulation...")
    # parameter
    Nx = 100
    Ny = 50
    # in physical units
    Lx = 10.0
    Ly = 10.0
    T_top = 1.0
    T_bottom = 10.0
    # time in internal units
    dt = 1e-5
    tspan = 1e-2
    # in physical units:
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
    @show Ra, Ra < 1700
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
    temperature[1:div(Nx, 2), end - 1] .= temperature[1, end]

    div_v = zeros(Nx, Ny)
    pressure = ones(Nx, Ny)

    explicit_velocity_rhs = similar(velocity)
    explicit_temperature_rhs = similar(temperature)

    return Sim(
        Nx, Ny, Lx, Ly, T_top, T_bottom, dt, tspan, mean_density, kinematic_viscosity,
        thermal_conductivity, heat_capacity_V, thermal_expansion_coeff,
        gravity_accel,
        C_1, C_2, C_3, Ra, T0,
        dx, dy, nsteps,
        velocity, velocity_without_pressure,
        temperature,
        div_v, pressure,
        explicit_velocity_rhs, explicit_temperature_rhs,
        0, 0.0,
    )
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

function time_step!(sim::Sim)
    # compute the explicit part
    println("explicit")
    for j in 2:sim.Ny-1
        for i in 1:sim.Nx
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
    eps = 1e-10
    prefactor_velocity = 1 / sim.dt + sim.C_1 * 2 * (1 / sim.dx^2 + 1 / sim.dy^2)
    copy!(sim.velocity_without_pressure, sim.velocity) # initial guess is the velocity from the last step
    while true
        mse = 0.0
        for j in 2:sim.Ny-1
            for i in 1:sim.Nx
                neighbor_sum = (
                    sim.velocity_without_pressure[mod1(i - 1, sim.Nx), j] / sim.dx^2 +
                    sim.velocity_without_pressure[mod1(i + 1, sim.Nx), j] / sim.dx^2 +
                    sim.velocity_without_pressure[i, j - 1] / sim.dy^2 +
                    sim.velocity_without_pressure[i, j + 1] / sim.dy^2
                )
                new_val = sim.explicit_velocity_rhs[i, j] + sim.C_1 * neighbor_sum
                new_val /= prefactor_velocity
                mse += norm(sim.velocity_without_pressure[i, j] - new_val)^2
                sim.velocity_without_pressure[i, j] = new_val
            end
        end
        mse = sqrt(mse / (sim.Nx * (sim.Ny - 2)))
        if !isfinite(mse)
            error("divergence")
        end
        @show mse
        if mse < eps
            break
        end
    end

    # gauss seidel - overrelaxation loop to solve the implicit equation for temperature
    println("implicit temperature")
    prefactor_temperature = 1 / sim.dt + sim.C_1 * 2 * (1 / sim.dx^2 + 1 / sim.dy^2)
    eps = 1e-10
    while true
        mse = 0.0
        for j in 2:sim.Ny-1
            for i in 1:sim.Nx
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
            end
        end
        mse = sqrt(mse / (sim.Nx * (sim.Ny - 2)))
        @show mse
        if mse < eps
            break
        end
    end
end

function chorin_projection!(sim::Sim)
    # compute the divergence of the updated velocity v*
    println("chorin_projection")
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
    # sim.div_v[:, :] ./= sim.dt

    # solve laplace P = density / dt * div v* using gauss-seidel iteration
    eps = 1e-6
    omega = 0.8
    iteration = 0
    while true
        mse = 0.0
        mean_P = 0.0
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
                mean_P += P_new^2
                sim.pressure[i, j] = P_new
            end
        end
        mean_P /= (sim.Nx * (sim.Ny - 2))
        # error calculation
        mse /= (sim.Nx * (sim.Ny - 2)) * mean_P
        mse = sqrt(mse)
        print("\rmse = $mse, mean_P = $mean_P, iteration = $iteration")
        iteration += 1
        # convergence
        if mse <= eps
            break
        end
    end
    println("")

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
            diff_x = (sim.pressure[mod1(i + 1, sim.Nx), j] - sim.pressure[mod1(i - 1, sim.Nx), j]) / (2*sim.dx)
            diff_y = (sim.pressure[i, j + 1]               - sim.pressure[i, j - 1])               / (2*sim.dy)
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
    println("plotting...")
    xs = range(0, sim.Lx, sim.Nx)
    ys = range(0, sim.Ly, sim.Ny)
    suptitle("thermal convection with boussinesq approximation and\nchorin projection, step = $(sim.step)")
    ncols = 2
    nrows = 2

    subplot(nrows, ncols, 1)
    pcolormesh(xs, ys, transpose(sim.temperature))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    gca().set_aspect("equal")
    title("temperature")

    subplot(nrows, ncols, 2)
    pcolormesh(xs, ys, transpose([v[1] for v in sim.velocity]))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    gca().set_aspect("equal")
    title("velocity x")

    subplot(nrows, ncols ,3)
    pcolormesh(xs, ys, transpose([v[2] for v in sim.velocity]))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    gca().set_aspect("equal")
    title("velocity y")

    subplot(nrows, ncols, 4)
    pcolormesh(xs, ys, transpose(sim.pressure))
    colorbar()
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    gca().set_aspect("equal")
    title("pressure")

    tight_layout()
end

end

sim = Convection.init()
for i in 1:sim.nsteps
    Convection.step!(sim)
end
