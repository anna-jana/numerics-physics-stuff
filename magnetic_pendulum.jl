using LinearAlgebra
using PyPlot
using OrdinaryDiffEq

function rhs(y, p, t)
    r, v = y[1:3], y[4:end]
    (M, g, mu0, m_pendulum, R, l, d, magnet_positions, m_magnet_vec) = p
    n = r ./ R
    m_pendulum_vec = m_pendulum .* n
    # forces from magnets (dipole - dipole interaction)
    F_dipol = zeros(3)
    for r_magnet in magnet_positions
        diff = r .- r_magnet
        dist = norm(diff)
        diff_hat = diff ./ dist
        prefactor = 3 * mu0 / (4*pi*dist^4)
        term1 = cross(cross(diff_hat, m_magnet_vec), m_pendulum_vec)
        term2 = cross(cross(diff_hat, m_pendulum_vec), m_magnet_vec)
        term3 = - 2 .* diff_hat .* dot(m_magnet_vec, m_pendulum_vec)
        term4 = 5 .* diff_hat .* dot(cross(diff_hat, m_magnet_vec),
                                      cross(diff_hat, m_pendulum_vec))
        F_dipol .+= prefactor .* (term1 .+ term2 .+ term3 .+ term4)
    end
    # force from gravity
    F_gravity = [0, 0, -g]
    F_real = F_dipol .+ F_gravity
    # constraint force which keeps the pendulum at fixed distance from origin
    F_constraint = - dot(F_real, n) .* n
    F_total = F_real .+ F_constraint
    return vcat(v, F_total ./ M)
end

M = 1.0
g = 1.0
mu0 = 1.
m_pendulum = 1.0
R = 1.
l = 0.5
d = 0.0
h = sqrt(3) / 6 * l
magnet_positions = [(0.0, 2 * h, d), (-l/2, - h, d), (+l/2, -h, d)]
m_magnet = 2.0
m_magnet_vec = m_magnet .* [0, 0, 1]
params = (M, g, mu0, m_pendulum, R, l, d, magnet_positions, m_magnet_vec)

tspan = 200.0
v0 = zeros(3)
phi0 = 0
theta0 = pi + pi / 8
r0 = [
    cos(phi0) * sin(theta0) * R,
    sin(phi0) * sin(theta0) * R,
    cos(theta0) * R
]

problem = ODEProblem(rhs, vcat(r0, v0), (0, tspan), params)
solution = solve(problem, Tsit5(), reltol=1e-10, saveat=tspan/1000)
x, y = solution[1, :], solution[2, :],

figure()
plot(x, y)
xlabel("x")
ylabel("y")
plot([0, -l/2, +l/2], [h, -h, -h], "o")
gca().set_aspect("equal")
show()
