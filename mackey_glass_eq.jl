using DelayDiffEq
using PyPlot
using DynamicalSystems
using Interpolations
using DelimitedFiles
using Printf
using LsqFit

################### simulation code for the mackey glass equation ##################
calc_dP(P, delayed_P, beta, n, gamma) = beta * delayed_P / (1 + delayed_P^n) - gamma * P

function mackey_glass_rhs(P, history, params, t)
  beta, n, gamma, tau = params
  delayed_P = history(params, t - tau)
  return calc_dP(P, delayed_P, beta, n, gamma)
end

const algorithm = MethodOfSteps(AutoTsit5(Rosenbrock23()))
const solve_kwargs = (abstol=1e-6, reltol=1e-6, maxiters=10^10)
const default_P0 = 0.5
const default_chaotic_params = (beta = 2, n = 9.65, gamma = 1, tau = 2)
const default_farmer_params = (beta = 0.2, n = 10.0, gamma = 0.1, tau = 16.8)
const farmer_taus = [17, 23, 23.8, 30]

function simulate_mackey_glass(beta, n, gamma, tau; tmax=100.0, t0=0.0, dt=0.1, P0=default_P0, dense=false)
    initial_history(p, t) = P0
    params = (beta, n, gamma, tau)
    prob = DDEProblem(mackey_glass_rhs, P0, initial_history, (t0, tmax), params; constant_lags=[tau])
    if dense
        return solve(prob, algorithm; solve_kwargs..., dense=true)
    else
        return solve(prob, algorithm; solve_kwargs..., saveat=dt)
    end
end

####################### some simple plots #######################
function plot_evolution()
    beta, n, gamma, _ = default_chaotic_params
    figure()
    subplot(2,1,1)
    tau = 1
    sol = simulate_mackey_glass(beta, n, gamma, tau)
    plot(sol.t, sol.u)
    xlabel("t")
    ylabel("P(t)")
    title("\$\\tau = $tau\$")

    subplot(2,1,2)
    tau = 2
    sol = simulate_mackey_glass(default_chaotic_params...)
    plot(sol.t, sol.u)
    xlabel("t")
    ylabel("P(t)")
    title("\$\\tau = $tau\$")

    suptitle("\$\\beta = $beta, n = $n, \\gamma = $gamma\$")
    tight_layout()
end

################## attractor of the mackey glass equation ######################
function mackey_glass_attractor(beta, n, gamma, tau;
                     embedding_delay=tau, do_3d=false,
                     periods=50, steps_per_period=100, transit_steps=0,
                     lw=1.5, do_plot=true, add_title=true)
    sol = simulate_mackey_glass(beta, n, gamma, tau;
                                   tmax=periods*tau,
                                   dt=embedding_delay / steps_per_period)
    if do_plot
        if do_3d
            plot3D(sol.u[1 + 2*steps_per_period + transit_steps:end],
                   sol.u[1 + steps_per_period + transit_steps:end-steps_per_period],
                   sol.u[1 + transit_steps:end-2*steps_per_period],
                   lw=lw)
        else
            plot(sol.u[1 + steps_per_period + transit_steps:end],
                 sol.u[1 + transit_steps:end-steps_per_period],
                 lw=lw)
        end
        xlabel("\$P(t)\$")
        ylabel("\$P(t - \\tau_\\mathrm{embedding})\$")
        if do_3d
            zlabel("\$P(t - 2\\tau_\\mathrm{embedding})\$")
        end
        if add_title
            title("\$\\beta = $beta, n = $n, \\gamma = $gamma, \\tau = $tau\$")
        end
    end
    return sol
end

function plot_tau_bifurcation()
    figure()
    rows, cols = 3, 2
    tau_list = [1.3, 1.4, 1.65, 1.71, 1.72, 1.74]
    beta, n, gamma, _ = default_chaotic_params
    for (i, tau) in enumerate(tau_list)
        subplot(rows, cols, i)
        mackey_glass_attractor(beta, n, gamma, tau;
                  add_title=false, lw=0.1, transit_steps=100*500, steps_per_period=100, periods=1000)
        title("\$\\tau = $tau\$")
    end
    suptitle("\$ \\beta = $beta, n = $n, \\gamma = $gamma, \\tau_\\mathrm{embedding} = \\tau\$")
    tight_layout()
end

plot_evolution()
plot_tau_bifurcation()
show()
