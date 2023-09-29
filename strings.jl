using FFTW
using PyPlot

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

log_start = 2.0
log_end = 3.0
k_max = 1.0

L = 1 / log_to_H(log_end)
N = ceil(Int, L * tau_to_a(log_to_tau(log_end)))
xs = range(0.0, stop=L, length=N)

tau_start = log_to_tau(log_start)
tau_end = log_to_tau(log_end)
tau_span = tau_end - tau_start

dx = L / N
hat = Array{Float64}(undef, (N, N))
ks = fftfreq(N, 1 / dx) .* (2*pi)
@inbounds for iy in 1:N
    @inbounds @simd for ix in 1:N
        k = sqrt(ks[ix]^2 + ks[iy]^2)
        hat[ix, iy] = k <= k_max ? (rand()*2 - 1) * field_max : 0.0
    end
end
field = ifft(hat)
theta = angle.(field)

@show (log_start, log_end)
@show (tau_start, tau_end)
@show L
@show N

figure()
pcolormesh(xs, xs, theta, cmap="twilight")
colorbar(label=raw"$\theta$")
xlabel("x")
ylabel("y")
title("Topological Strings")
show()
