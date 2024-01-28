using DynamicalSystems
using DelayDiffEq
using PyPlot
using Interpolations
using LinearAlgebra
using Statistics
using DelimitedFiles

function pyragas_rhs(state, history, params, time)
    chaotic_system, feedback_amplitude_vector, feedback_delay = params
    delayed_state = history(params, time - feedback_delay)
    F = feedback_amplitude_vector .* (delayed_state .- state)
    return chaotic_system.f(state, chaotic_system.p, time) .+ F
end

function pyragas_control(chaotic_system, feedback_amplitude_vector, feedback_delay, tmax;
        dt=0.1, system_transit=0.0, dense=false, ode_algorithm=AutoTsit5(Rosenbrock23()),
        solve_kwargs=(abstol=1e-6, reltol=1e-3, maxiters=10^10))
    ode_problem = ODEProblem(chaotic_system, (- feedback_delay - system_transit, 0.0))
    initial_solution = solve(ode_problem, ode_algorithm; dense=true, solve_kwargs...)
    initial_history(p, t) = initial_solution(t)
    initial_state = initial_solution.u[end]
    params = (chaotic_system, feedback_amplitude_vector, feedback_delay)
    problem = DDEProblem(pyragas_rhs, initial_state, initial_history, (0.0, tmax), params; constant_lags=[feedback_delay])
    dde_algorithm = MethodOfSteps(ode_algorithm)
    return solve(problem, dde_algorithm; dense=dense, saveat=dense ? [] : dt, solve_kwargs...)
end

const sys = Systems.roessler()
const system_transit = 1000.0
const v = SVector(0.0, 1.0, 0.0)
const default_K = 0.2
const default_tau = 17.5

function plot_control_example()
    K = default_K
    tau = default_tau
    tmax = 500.0
    t_chaotic = trajectory(sys, tmax; Ttr=system_transit)[1]
    feedback_amplitude_vector = K.*v
    t_control = pyragas_control(sys, feedback_amplitude_vector, tau, tmax; system_transit=system_transit)
    dt = t_control.t[2] - t_control.t[1]
    start = ceil(Int, tau/dt)
    transit_end = 1 + ceil(Int, 480.0/dt)
    F = [dot(feedback_amplitude_vector .* (t_control(t - tau) .- t_control(t)), v) for t in t_control.t[start:end]]

    figure()
    subplot(2,1,1)
    plot(t_chaotic[:, 1], t_chaotic[:, 2], lw=0.5, label="chaos")
    plot(t_control[1, transit_end:end], t_control[2, transit_end:end], lw=2.0, label="control")
    legend(framealpha=1.0)
    xlabel("x")
    ylabel("y")
    subplot(2,1,2)
    plot(t_control.t[start:end], F)
    xlabel("t")
    ylabel("F")
    suptitle("Pyragas Control of the Rossler Oscillator")
    tight_layout()
end

plot_control_example()
show()
