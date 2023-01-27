using OrdinaryDiffEq, PyPlot, LinearAlgebra

const tol = 1e-8

function compute_max_lyapunov_exponent(rhs_fn, x0, params, δ0, Δt, N)
    # solve the reference trajectory
    problem = ODEProblem(rhs_fn, x0, (0, N*Δt), params)
    solver = AutoTsit5(Rosenbrock23())
    ref_sol = solve(problem, solver, reltol=tol, abstol=tol, dense=true)
    # neighboring trajectory
    du = ref_sol.u[2] - ref_sol.u[1] # vector along trajectory
    v = rand(size(du, 1))
    pert = v - dot(du, v) / dot(du, du) * du # vector orthogonal to trajectory
    pert *= δ0 / norm(pert) # make pertubation small
    x_test = x0 + pert
    # result
    s = 0.0
    for i in 1:N
        t = i*Δt
        # advance neighboring trajectory
        p = ODEProblem(rhs_fn, x_test, ((i - 1)*Δt, t), params)
        neighboring_sol = solve(p, solver,  reltol=tol, abstol=tol, dense=true)
        # compute seperation
        x_ref = ref_sol(t)
        x_pert = neighboring_sol(t)
        diff = x_pert - x_ref
        δ = norm(diff)
        # update result
        s += log(δ / δ0)
        # rescale the neighboring trajectory
        x_test = x_ref + δ0 / δ * diff
    end
    # result
    return s / (Δt * N)
end

function lorentz_rhs(du, u, p, t)
    β, σ, ρ = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = - x * z - ρ * x - y
    du[3] = x * y - β * z
end

β = 8 / 3.
σ = 10.0
ρ_range = range(20, 100, length=30)
λs = [compute_max_lyapunov_exponent(lorentz_rhs, [1.0, 1.0, 1.0], [β, σ, ρ], 1e-9, 1e-2, 20000) for ρ = ρ_range]
plot(ρ_range, λs)
xlabel("\$\\rho\$")
ylabel("\$\\lambda_1\$")
