using FFTW
using PyPlot
using Random

log_to_H(l) = 1.0 / exp(l)
H_to_t(H) = 1 / (2*H)
t_to_H(t) = 1 / (2*t)
H_to_log(H) = log(1/H)
t_to_tau(t) = -2*sqrt(t)
log_to_tau(log) = t_to_tau(H_to_t(log_to_H(log)))
t_to_a(t) = sqrt(t)
tau_to_t(tau) = -0.5*(tau)^2
tau_to_a(tau) = -0.5*tau
tau_to_log(tau) = H_to_log(t_to_H(tau_to_t(tau)))

const field_max = 1 / sqrt(2)
const k_max = 1.0

function random_field(N, dx)
    hat = Array{Float64}(undef, (N, N))
    ks = fftfreq(N, 1 / dx) .* (2*pi)
    @inbounds for iy in 1:N
        @inbounds @simd for ix in 1:N
            k = sqrt(ks[ix]^2 + ks[iy]^2)
            hat[ix, iy] = k <= k_max ? (rand()*2 - 1) * field_max : 0.0
        end
    end
    field = ifft(hat)
    return field
end

function compute_dot_dot!(force, field, tau, dx)
    a = tau_to_a(tau)
    N = size(field, 1)
    @inbounds for j in 1:N
        for i in 1:N
            laplace = (field[mod1(i - 1, N), j] + field[mod1(i + 1, N), j] +
                       field[i, mod1(j - 1, N)] + field[i, mod1(j - 1, N)]
                       - 4 * field[i, j]) / dx^2
            force[i, j] = laplace - field[i, j] * (abs2(field[i, j]) - 0.5*a)
        end
    end
end

function run()
    Random.seed!(322313)
    nsteps = 1000
    log_start = 2.0
    log_end = 3.0

    L = 1 / log_to_H(log_end)
    N = ceil(Int, L * tau_to_a(log_to_tau(log_end)))
    xs = range(0.0, stop=L, length=N)
    tau_start = log_to_tau(log_start)
    tau_end = log_to_tau(log_end)
    tau_span = tau_end - tau_start
    dtau = tau_span / nsteps
    dx = L / N

    @show (log_start, log_end)
    @show (tau_start, tau_end)
    @show L
    @show N

    field = random_field(N, dx)
    field_dot = random_field(N, dx)
    field_dot_dot = similar(field)
    field_dot_dot_new = similar(field)
    field0 = copy(field)

    tau = tau_start

    compute_dot_dot!(field_dot_dot, field, tau, dx)

    for i in 1:nsteps
        println("step = $i / $nsteps")
        @. field += dtau * field_dot + dtau^2/2 * field_dot_dot
        compute_dot_dot!(field_dot_dot_new, field, tau, dx)
        @. field_dot += dtau / 2 * (field_dot_dot + field_dot_dot_new)
        field_dot_dot = field_dot_dot_new
        tau += dtau
    end

    figure()
    theta = angle.(field)
    theta0 = angle.(field0)
    subplot(1,2,1)
    pcolormesh(xs, xs, theta, cmap="twilight")
    title("initial tau = $tau_start")
    colorbar(label=raw"$\theta$")
    xlabel("x")
    ylabel("y")
    subplot(1,2,2)
    pcolormesh(xs, xs, theta0, cmap="twilight")
    title("final tau = $tau_end")
    colorbar(label=raw"$\theta$")
    xlabel("x")
    ylabel("y")
    tight_layout()
    show()
end

run()
