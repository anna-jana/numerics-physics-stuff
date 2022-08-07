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
    return solve(problem, dde_algorithm; solve_kwargs, dense=dense, saveat=dense ? [] : dt)
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
    t_chaotic = trajectory(sys, tmax; Ttr=system_transit)
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
    savefig("pyragas_control_of_roessler_attractor.pdf")
end

const disp_disp_tau_filename = "pyragas_control_displacement_dispersions_tau.dat"
const disp_disp_K_filename = "pyragas_control_displacement_dispersions_K.dat"

function compute_dispersions_tau()
    tmax = 5000.0
    nsamples = 1000
    control_transit = 1000.0
    taus = range(4.0, 20.0, length=500)
    @time displacement_dispersions = map(taus) do tau
        sol = pyragas_control(sys, default_K.*v, tau, tmax; system_transit=system_transit, dense=true)
        mean((sol(t - tau)[2] - sol(t)[2])^2 for t = tau + control_transit:tmax/nsamples:tmax)
    end
    writedlm(disp_disp_tau_filename, [taus displacement_dispersions])
end

const disp_tau = 5.9

function compute_dispersions_K()
    tmax = 5000.0
    nsamples = 1000
    control_transit = 1000.0
    Ks = range(0.0, 1.1, length=500)
    tau = disp_tau
    @time displacement_dispersions = map(Ks) do K
        sol = pyragas_control(sys, K.*v, tau, tmax; system_transit=system_transit, dense=true)
        mean((sol(t - tau)[2] - sol(t)[2])^2 for t = tau + control_transit:tmax/nsamples:tmax)
    end
    writedlm(disp_disp_K_filename, [Ks displacement_dispersions])
end

function plot_dispersions()
    figure()

    data = readdlm(disp_disp_tau_filename)
    taus, displacement_dispersions = data[:, 1], data[:, 2]
    subplot(2,1,1)
    plot(taus, displacement_dispersions, ".")
    xlabel(raw"$\tau$")
    ylabel(raw"$\langle D(t)^2 \rangle$")
    yscale("log")
    ylim(1e-11, 1e2)
    title("K = $default_K")

    data = readdlm(disp_disp_K_filename)
    Ks, displacement_dispersions = data[:, 1], data[:, 2]
    subplot(2,1,2)
    plot(Ks, displacement_dispersions, ".")
    xlabel("K")
    ylabel(raw"$\langle D(t)^2 \rangle$")
    yscale("log")
    # ylim(1e-11, 1e2)
    title("\$\\tau\$ = $disp_tau")

    tight_layout()
    savefig("pyragas_control_displacement_dispersion_plot.pdf")
end

const max_lyap_filename = "pyragas_max_lyap.dat"

function pyragas_map!(next_history_vector, _initial_history_vector, params, _t)
    chaotic_system, feedback_amplitude_vector, feedback_delay, N = params

    initial_history_vector = [SVector{N}(_initial_history_vector[i:i+N-1]) for i = 1:N:length(_initial_history_vector) - N]

    dt = feedback_delay / (length(initial_history_vector) - 1)
    initial_state = initial_history_vector[end]
    interp = LinearInterpolation(range(-feedback_delay, 0.0, length=length(initial_history_vector)), initial_history_vector)
    initial_history(p, t) = interp(t)
    problem = DDEProblem(pyragas_rhs, initial_state, initial_history, (0.0, feedback_delay + dt), params; constant_lags=[feedback_delay])

    dde_algorithm = MethodOfSteps(AutoTsit5(Rosenbrock23()))
    sol = solve(problem, dde_algorithm; saveat=dt)

    for (i, u) in enumerate(@view(sol.u[2:end]))
        copy!(@view(next_history_vector[(i - 1)*N + 1 : i*N]), u)
    end
    return nothing
end

function max_lyapunov(K, tau; N=80)
    dt = tau / (N - 1)
    sol = pyragas_control(sys, K.*v, tau, tau; system_transit=1000.0, dt=dt)
    ds = DiscreteDynamicalSystem(pyragas_map!, vcat(sol.u...), [sys, K.*v, tau, length(sys.u0)])
    return lyapunov(ds, 100; Ttr=round(Int, 500.0/(tau + dt)), show_progress=true)
end

function compute_max_lyapunov_exponent()
    Ks = range(0.0, 1.1, length=10)
    @time lambda_maxs = map(K -> max_lyapunov(K, disp_tau), Ks)
    writedlm(max_lyap_filename, [Ks lambda_maxs])
end

function plot_max_lyapunov_exponent()
    data = readdlm(max_lyap_filename)
    Ks, lambda_maxs = data[:, 1], data[:, 2]
    figure()
    plot(Ks, lambda_maxs, ".")
    axhline(0.0, color="black")
    xlabel("K")
    ylabel(raw"$\lambda_\mathrm{max}$")
    savefig("pyragas_max_lyapunov_exponent.pdf")
end

function main()
    plot_control_example()
    compute_dispersions_tau()
    compute_dispersions_K()
    plot_dispersions()
    compute_max_lyapunov_exponent()
    plot_max_lyapunov_exponent()
end
