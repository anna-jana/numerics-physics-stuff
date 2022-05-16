using DifferentialEquations
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
    savefig("machey_glass_evolution.pdf")
end

function plot_production_feedback_function()
    figure()
    x = range(0, 2, length=400)
    cmap = PyPlot.cm_get_cmap("viridis")
    n_list = [3, 5, 7, 10, 15, 20]
    for (i, n) in enumerate(n_list)
        plot(x, @.(x / (1 + x^n)), color=cmap((i - 1)/(length(n_list) - 1)),
             label="n = $n")
    end
    xlabel("x")
    ylabel("\$ x/(1 + x^n) \$")
    legend(ncol=2, framealpha=1)
    title("\$\\beta = 1\$")
    savefig("mackey_glass_feedback_function.pdf")
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

########################### bifurcations of the macky glass ###########################
# n_list = [7, 7.75, 8.5, 8.79, 9.65, 9.696, 9.7056, 9.7451, 10, 20] # numbers from the scholarpedia page
# beta_list = [1.5, 1.7, 1.76, 1.782, 1.8, 2.0, 2.1]
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
    savefig("mackey_glass_tau_period_doubling_bifurcation.pdf")
end

############################### poincare section of the macky glass eq #############################
include("psos_from_fn.jl")

const farmer_delay = 10

# from the FARMER paper
function mackey_glass_poincare_section(beta, n, gamma, tau;
        crossing_val=0.85, delay=farmer_delay, direction=-1,
        time=10000, transit=100, bisections=time)
    tmax = delay * time
    @time sol = simulate_mackey_glass(beta, n, gamma, tau; tmax=tmax, dense=true)
    f(t) = SVector(sol(t), sol(t - delay), sol(t - 2*delay))
    @time psos = poincare_section_from_function(f, transit*delay, tmax, bisections,
            SVector(crossing_val, 0, 0), SVector(sign(direction), 0, 0))
    return psos
end

fractal_dimension(psos) = generalized_dim(psos; q=0.0)

psos_filename(tau) = "mackey_glass_psos_tau=$tau.dat"

function compute_poincare_sections()
    for tau in farmer_taus
        psos = mackey_glass_poincare_section(
            default_farmer_params.beta, default_farmer_params.n,
            default_farmer_params.gamma, tau;
            time=1_000_000, transit=500)
        writedlm(psos_filename(tau), psos)
    end
end

const psos_dim_filename = "mackey_glass_psos_fractal_dimension.dat"

function process_poincare_sections()
    Ds = Float64[]
    fig, axes = subplots(2, 2)
    for (i, tau) in enumerate(farmer_taus)
        psos = readdlm(psos_filename(tau))
        axes[i].plot(psos[:, 2], psos[:, 3], ".", ms=0.1)
        axes[i].set_xlabel("P(t - $farmer_delay)")
        axes[i].set_ylabel("(P(t - $(2*farmer_delay))")
        D = fractal_dimension(standardize(Dataset(@view(psos[:, 2:3]))))
        push!(Ds, D)
        axes[i].set_title(@sprintf("\$\\tau\$ = %.2f, D = %.2f", tau, D))
    end
    tight_layout()
    savefig("mackey_glass_psos_plot.png")
    writedlm(psos_dim_filename, Ds)
end

########################### lyapunov exponents of the macky glass eq ############################
# advance history vector by tau + dt
function mackey_glass_method_of_steps_map!(next_history_vector, initial_history_vector, params, _t)
    beta, n, gamma, tau = params
    dt = tau / (length(initial_history_vector) - 1)
    interp = LinearInterpolation(range(0, tau, length=length(initial_history_vector)), initial_history_vector)
    initial_history(p, t) = interp(t + tau)
    problem = DDEProblem(mackey_glass_rhs, initial_history_vector[end], initial_history, (0.0, tau + dt), params; constant_lags=[tau])
    solution = solve(problem, algorithm; solve_kwargs..., saveat=dt)
    next_history_vector[1:end] = @view(solution.u[2:end])
    return nothing
end

function mackey_glass_lyapunov_spectrum(beta, n, gamma, tau; N=80, transit=50, L=50)
    dds = DiscreteDynamicalSystem(mackey_glass_method_of_steps_map!, repeat([default_P0], N), (beta, n, gamma, tau))
    return lyapunovspectrum(dds, L; Ttr=transit, show_progress=true)
end

function lyapunov_dimension(spectrum)
    if spectrum[1] <= 0.0
        return NaN
    else
        j = findfirst(spectrum .< 0.0)
        @assert j < length(spectrum)
        return j + sum(spectrum[1:j]) / abs(spectrum[j + 1])
    end
end

const lyapunov_filename = "mackey_glass_lyapunov_spectra.dat"

function compute_lyapunov_spectra()
    beta, n, gamma, _ = default_farmer_params
    N = 30
    specs = Matrix{Float64}(undef, N, length(farmer_taus))
    for (i, tau) in enumerate(farmer_taus)
        specs[:, i] = mackey_glass_lyapunov_spectrum(beta, n, gamma, tau; N=N, transit=50, L=30)
    end
    writedlm(lyapunov_filename, specs)
end

const lyapunov_dim_filename = "mackey_glass_lyapunov_dim.dat"

function process_lyapunov_spectra()
    specs = readdlm(lyapunov_filename)
    figure()
    Ds = Float64[]
    for (i, tau) in enumerate(farmer_taus)
        spectrum = specs[:, i]
        D = lyapunov_dimension(spectrum)
        push!(Ds, D)
        step(1:length(spectrum), spectrum, where="mid", label="\$\\tau = $tau\$")
    end
    xlabel("i")
    ylabel("\$\\lambda_i\$")
    beta, n, gamma, _ = default_farmer_params
    title("\$\\beta = $beta, n = $n, \\gamma = $gamma\$")
    legend()
    savefig("mackey_glass_lyapunov_spectra_plot.pdf")
    writedlm(lyapunov_dim_filename, Ds)
end

function plot_dimensions()
    psos_dims = readdlm(psos_dim_filename)
    lyapunov_dims = readdlm(lyapunov_dim_filename)
    figure()
    plot(farmer_taus, psos_dims .+ 1, "x", label="Dimension from Poincare Section")
    plot(farmer_taus, lyapunov_dims, "+", label="Dimension from Lyapunov Spectrum")
    xlabel(raw"Delay, $\tau$")
    ylabel("Dimension, D")
    legend()
    savefig("mackey_glass_dim_plot.pdf")
end

############################################# main ###############################################
function main()
    plot_production_feedback_function()
    plot_evolution()
    plot_tau_bifurcation()
    compute_poincare_sections()
    process_poincare_sections()
    compute_lyapunov_spectra()
    process_lyapunov_spectra()
    plot_dimensions()
end


function test_fractal_dim(i; tol=0.25, do_standardize=false, k=20, w=1, z=-1)
    psos = Dataset(@view(readdlm(psos_filename(farmer_taus[i]))[:, 2:3]))
    if do_standardize
        psos = standardize(psos)
    end
    bs = estimate_boxsizes(psos; k=k, w=w, z=z)
    H = genentropy.(Ref(psos), bs; q=0.0)
    x = - log.(bs)
    idx, slops = linear_regions(x, H; tol=tol)
    for i in 1:length(idx)-1
        r = idx[i] + 1 : idx[i+1]
        plot(x[r], H[r], "x")
    end
    lr = findmax([idx[i+1] - idx[i] for i=1:length(idx)-1])[2]
    r = idx[lr] + 1 : idx[lr + 1]
    fit = curve_fit((x, p) -> @.(p[1]*x + p[2]), x[r], H[r], [1.0, 1.0])
    slope = fit.param[1]
    slope_error = sqrt(estimate_covar(fit)[1, 1])
    plot(x[r], @.(fit.param[2] + slope*x[r]), "-k")
    return slope, slope_error
end
