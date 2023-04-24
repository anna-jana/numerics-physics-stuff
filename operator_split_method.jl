using PyPlot
using FFTW

function simulate_1d_schroedinger(V_fn, psi0_fn, m;
        L = 1.0, N = 1000, tspan = 10.0, dt = 0.01, save_interval = 5.0,
    )
    # grid and stepping
    dx = L / N # periodic boundary conditions
    x = 0:dx:(L-dx)
    nsteps = ceil(Int, tspan / dt)
    k = 2*pi .* fftfreq(N, 1/dx)

    # initial condition
    psi0 = psi0_fn.(x)
    psi0 = Complex.(psi0 ./ (sum(psi0) * dx))
    psi_hat = fft(psi0)

    # propagator
    V = V_fn.(x)
    kin_step = exp.(- 1im * dt / 2 .* k.^2 ./ (2*m))
    pot_step = exp.(- 1im * dt .* V)

    # time stepping loop
    solutions = Tuple{Float64, Vector{Complex{Float64}}}[]
    save_step = floor(Int, save_interval / dt)
    for i in 0:nsteps
        if i % save_step == 0
            push!(solutions, (i * dt, ifft(psi_hat)))
        end
        psi_hat = kin_step .* fft(pot_step .* ifft(kin_step .* psi_hat))
    end

    # plotting
    figure(layout="constrained")
    subplot(2,1,1)
    plot(x, V)
    xlabel("x")
    ylabel("V(x)")
    subplot(2,1,2)
    for (t, psi) in solutions
        plot(x, abs2.(psi), label="t = $t")
    end
    legend()
    xlabel("x")
    ylabel(raw"$|\psi(t, x)|$")
    show()

    return solutions
end


V0 = 100.0
sigma = 0.001
omega = 0.01
L = 1.0
x0 = 0.5 * L
n = 3
V_fn(x) = V0 * cos(2*pi*n*x/L)
psi0_fn(x) = exp(- (x - x0)^2 / (2*sigma)) * exp(- omega * 1im * (x - x0))
simulate_1d_schroedinger(V_fn, psi0_fn, 1.0; L=L)

