using PyPlot
using OrdinaryDiffEq
using Random
using FFTW

@inline f(v) = - v^3 + v

function allen_cahn_rhs(dv, v, p, _)
    eps2_inv, dx2_inv, dy2_inv = p
    Nx, Ny = size(v)
    @inbounds for j in 1:Ny, i in 1:Nx
        d2vdx2 = (v[mod1(i - 1, Nx), j] - 2*v[i, j] + v[mod1(i + 1, Nx), j]) * dx2_inv
        d2vdy2 = (v[i, mod1(j - 1, Ny)] - 2*v[i, j] + v[i, mod1(j + 1, Ny)]) * dy2_inv
        dv[i, j] = eps * (d2vdx2 + d2vdy2) + f(v[i, j])
    end
end

Nx = 40
Ny = 40
Lx = 1.0
Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny
eps = 0.001
nsteps = 1000
tmax = 10.0

Random.seed!(42)
kx = fftfreq(Nx, 1 / dx)
ky = fftfreq(Ny, 1 / dy)
kmax = sqrt(maximum(abs.(kx))^2 + maximum(abs.(ky))^2) / 5.0
mode = ComplexF64[
     sqrt(kx[i]^2 + ky[j]^2) < kmax ? rand() * exp(rand()*2*pi*im) : 0.0
    for j in 1:length(ky), i in 1:length(kx)]
v0 = real.(ifft(mode))

problem = ODEProblem(allen_cahn_rhs, v0, (0, tmax), (1 / eps^2, 1 / dx^2, 1 / dy^2))
solver = TRBDF2()
solution = solve(problem, solver, saveat=tmax / nsteps)

figure()
xs = range(0, Lx, Nx)
ys = range(0, Ly, Ny)
for u in solution.u
    clf()
    pcolormesh(xs, ys, u, vmin=minimum(v0), vmax=maximum(v0))
    xlabel("x")
    ylabel("y")
    title("Allen Cahn Equation")
    colorbar(label="v(x,y)")
    pause(0.0001)
end
