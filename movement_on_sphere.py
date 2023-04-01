import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

sp.init_printing()

r, m, g = sp.symbols("r, m, g", real=True)
t = sp.symbols("t", real=True)
phi_fn = sp.Function("phi", real=True)
theta_fn = sp.Function("theta", real=True)
phi = phi_fn(t)
theta = theta_fn(t)
phi_dot = phi.diff(t)
theta_dot = theta.diff(t)
phi_dotdot = phi.diff(t, 2)
theta_dotdot = theta.diff(t, 2)

x = r * sp.cos(phi) * sp.sin(theta)
y = r * sp.sin(phi) * sp.sin(theta)
z = r * sp.cos(theta)
x_dot = x.diff(t)
y_dot = y.diff(t)
z_dot = z.diff(t)
T = (x_dot**2 + y_dot**2 + z_dot**2) * m / 2
V = g * z
L = T - V

lagrange_phi = sp.Eq(L.diff(phi_dot).diff(t), L.diff(phi)).simplify()
lagrange_theta = sp.Eq(L.diff(theta_dot).diff(t), L.diff(theta)).simplify()
[rhs_phi_dot_dot] = sp.solve(lagrange_phi, phi_dotdot)
[rhs_theta_dot_dot] = sp.solve(lagrange_theta, theta_dotdot)

sp.pprint(lagrange_phi)
sp.pprint(lagrange_theta)

def make_fn(expr):
    return sp.lambdify((phi, theta, phi_dot, theta_dot, r, m, g),
            expr, "numpy",)
rhs_phi_dot_dot_fn = make_fn(rhs_phi_dot_dot)
rhs_theta_dot_dot_fn = make_fn(rhs_theta_dot_dot)

def rhs(t, u, r, m, g):
    phi, theta, phi_dot, theta_dot = u
    phi_dot_dot = rhs_phi_dot_dot_fn(
        phi, theta, phi_dot, theta_dot, r, m, g)
    theta_dot_dot = rhs_theta_dot_dot_fn(
        phi, theta, phi_dot, theta_dot, r, m, g)
    return np.array([phi_dot, theta_dot, phi_dot_dot, theta_dot_dot])

tspan = 100.0
r = 1.0
m = 1.0
g = 1.0
theta0 = np.pi / 2
phi0_dot = 0.1
theta0_dot = 0.0

sol = solve_ivp(rhs, (0, tspan),
                np.array([0.0, theta0, phi0_dot, theta0_dot]),
                args=(r, m, g), dense_output=True)

n_points = 500
ts = np.linspace(0, tspan, n_points)
phi, theta, phi_dot, theta_dot = sol.sol(ts)

def plot_sphere_curve(ax, phi, theta, r):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    ax.plot(x, y, z, color="blue")

def plot_sphere(ax, r, n_phi=50, n_theta=50):
    phi, theta = np.meshgrid(
        np.linspace(0, 2 * np.pi, n_phi),
        np.linspace(0, np.pi, n_theta),
    )
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    ax.plot_surface(x, y, z, alpha=0.3)
    ax.scatter([0], [0], [r], color="red")
    ax.scatter([0], [0], [-r], color="green")

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
plot_sphere(ax, r)
plot_sphere_curve(ax, phi, theta, r)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.show()
