# Diffusion Quantum Monte Carlo for a single particle
# https://arxiv.org/pdf/physics/9702023.pdf
#
# hbar = 1

using PyPlot
using StatsBase

function dqmc(
    m, V_fn;
    nwalkers0 = 50000,
    x0 = 0.0,
    Delta_tau = 0.01,
    N_replicate_max = 3,
    E_R_0 = 1.0,
    nsteps = 1000,
)
    sigma = sqrt(Delta_tau / m)
    xs = repeat([x0], nwalkers0)
    new_xs = copy(xs)
    E_R = E_R_0

    for i in 1:nsteps
        empty!(new_xs)
        for x in xs
            x_new = x + sigma * randn()
            V_x = V_fn(x_new)
            W = exp(- (V_x - E_R) * Delta_tau)
            N_replicate = min(floor(Int, W + rand()), N_replicate_max)
            for j in 1:N_replicate
                push!(new_xs, x_new)
            end
        end
        N_prev = size(xs, 1)
        N_next = size(new_xs, 1)
        E_R = E_R + 1 / Delta_tau * (1 - N_next / N_prev)
        xs, new_xs = new_xs, xs
    end

    return E_R, xs
end

function main()
    m = 1
    k = 1
    E_0, xs = dqmc(m, x -> 0.5*k*x^2)
    nbins = 50
    x = range(minimum(xs), maximum(xs), length=nbins + 1)
    bin_width = x[2] - x[1]
    h = fit(Histogram, xs, x, closed=:left)
    h.weights[end] += 1 # include the right most value which is on the open bin edge
    w = h.weights
    c = w / sqrt(sum(w.^2) * bin_width)
    plot(x[1:end-1] .+ bin_width/2, c; label="DQMC")
    omega = sqrt(k/m)
    s =  @. (m*omega/pi)^(1/4)*exp(-m*omega*x^2/2)
    plot(x, s; label="analytic")
    legend()
    xlabel("x")
    ylabel(raw"$\psi_0$(x)")
    title(raw"$E_0$ = 1/2 vs " * "$E_0")
    show()
end
main()
