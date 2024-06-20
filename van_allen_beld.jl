using LinearAlgebra
using PyPlot
using OrdinaryDiffEq

function compute_magnetic_dipole_field(x, m)
    mu0 = 1.0
    r = norm(x)
    return mu0 / (4*pi) * (3 * dot(m, x) * x / r^5 - m / r^3)
end

function rhs(y, p, t)
    q, m, m_earth = p
    x, v = y[1:3], y[4:end]
    B = compute_magnetic_dipole_field(x, m_earth)
    a = q / m * cross(v, B)
    return vcat(v, a)
end

v_parallel_0 = 1e-8
v_orthogonal_0 = 1e-6
M = 10.0
m_earth = [0.0, 0.0, M]
q = 1.0
m = 1.0
L = 5.0
tmax = 1e7
x0 = [L, 0.0, 0.0,]
v0 = [0.0, v_orthogonal_0, v_parallel_0]

problem = ODEProblem(rhs, vcat(x0, v0), (0, tmax), (q, m, m_earth))
solver = Tsit5()
solution = solve(problem, solver, reltol=1e-6, abstol=1e-10)

figure()
a = 1
for i in 1:3
    for j in 1:3
        subplot(3, 3, a)
        plot(solution[i, 1:100:end], solution[j, 1:100:end], lw=1)
        xlabel(["x", "y", "z"][i])
        ylabel(["x", "y", "z"][j])
        global a += 1
    end
end
tight_layout()

figure()
plot3D(solution[1, 1:500], solution[2, 1:500], solution[3, 1:500])
xlabel("x")
ylabel("y")
zlabel("z")

show()

