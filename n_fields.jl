using OrdinaryDiffEq
using PyPlot
using Random

H_to_t(H) = 1 / (2*H)
t_to_H(t) = 1 / (2*t)

function rhs(du, u, p, t)
    H = H_to_t(t)
    N, M, G = p
    @views phis = u[1:N]
    @views phi_dots = u[N+1:end]
    @views du[1:N] = u[N + 1 : end]
    for i in 1:N
        du[N + i] = - 3 * H * phi_dots[i] - M[i]^2 * phis[i]
        for j in 1:N
            du[N + i] += - G[i, j] * phis[i] * phis[j]^2
        end
    end
end

function compute_number_density(u, p)
    N, M, G = p
    return sum(1:N) do i
        E = 0.5 * u[N + i]^2 + M[i]^2 / 2 * u[i]^2
        E += sum(G[i, j] * u[i]^2 * u[j]^2 for j in 1:N)
        return E / M[i]
    end
end

H0 = 100.0
H1 = 1e-5
t0 = H_to_t(H0)
t1 = H_to_t(H1)
Random.seed!(2141243)
N = 20
phi0 = randn(N)
phi_dot0 = zeros(N)
M = randn(N).^2
G = randn(N, N).^2
params = (N, M, G)

problem = ODEProblem(rhs, vcat(phi0, phi_dot0), (t0, t1), params)
solver = AutoTsit5(Rosenbrock23())
solution = solve(problem, solver, reltol=1e-10, abstol=1e-10)

n = [compute_number_density(solution[:,i], params) for i in 1:size(solution.u, 1)]
H = t_to_H.(solution.t)
s = H.^(3/2)

figure()
loglog(H, n ./ s)
xlabel("H")
ylabel("total comoving number density")
gca().invert_xaxis()
show()
