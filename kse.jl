using OrdinaryDiffEq, FFTW, LinearAlgebra, PyPlot

# rhs for the kuramoto - sivashinsky equation
function kse_spectral_rhs!(dy, y, p, t)
    forward_rfft_plan, inverse_rfft_plan, u, sq, A, B = p
    # calc F[u^2] using fft(u^2) = fft(ifft(y)^2)
    # (avoid allocating memory and using the plans)
    mul!(u, inverse_rfft_plan, y)
    u .*= u
    mul!(sq, forward_rfft_plan, u)
    # equation in fourier space:
    @. dy = sq*A + B*y
end

function plot_kse()
    # parameter
    b = 30.0
    dx = 0.4
    dt = 0.01
    tmax = 100.0

    # precompute and preallocate things
    # spatial discretisation
    x = 0:dx:b

    # initial condition
    u0 = @. cos((2*pi/b)*x)

    # fouier transform stuff
    ks = collect(FFTW.rfftfreq(length(u0), 1/dx)) * 2*pi
    forward_rfft_plan = FFTW.plan_rfft(u0)
    y0 = forward_rfft_plan * u0
    inverse_rfft_plan = FFTW.plan_irfft(y0, length(u0))

    # constant coefficients
    A = @. im*ks/2.0
    C = @. im*ks
    B = @. ks^2 - ks^4

    # temporal memory
    u = similar(u0)
    sq = similar(y0)

    # parameters
    p = (forward_rfft_plan, inverse_rfft_plan, u, sq, A, B, C)

    # solve the ode problem with a library solver
    t = 0.0:dt:tmax
    ode_problem = ODEProblem(kse_spectral_rhs!, y0, (0.0, tmax), p)
    @time solution = solve(ode_problem, Tsit5(); saveat=t, maxiters=10^10)
    u = [inverse_rfft_plan * y for y in solution.u]

    # plot the solution
    pcolormesh(x, t, u, cmap="BrBG", shading="nearest")
    xlabel("x")
    ylabel("t")
    colorbar(label="u")
    title("kuramoto - sivashinsky equation")
    tight_layout()
end

plot_kse()
