import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as a

# c^2 d^2/dx^2 phi = d^2/dt^2 phi
#
# phi = f(x - c*t) + g(x + c*t)
# d phi / dt = - c * f(x - c*t) + c * g(x + c*t)
#
# let c = 1
#
# phi = f(x - t) + g(x + t)
# d phi / dt = - f'(x - t) + g'(x + t)
# d phi / dx = f'(x - t) + g'(x + t)
#
# f'(x) = (d phi0 / dx(x) + phi0 dot(x)) / 2
# g'(x) = (d phi0 / dx(x) - phi0 dot(x)) / 2
#
# dirichet:
# phi(x=0) = 0
# phi = f(-t) + g(t) = 0
# f(t) = - g(-t) => right and left moving waves cancel at x = 0
#
# von neumann:
# d phi / dx (x = 0) = 0
# f'(-t) + g'(t) = 0
# g'(t) = -f'(-t)
# d/dx g(x) = d/dx f(-t)
# choose constant of integration as C = 0
# g(t) = f(-t)

def plot_wave_eq_1d_solution(name, f, g):
    n = 500
    L = 1.0
    xs = np.linspace(0, L, n)

    def phi(t):
        return f(xs - t) + g(xs + t)

    nsteps = 100
    tspan = 2.0
    dt = tspan / nsteps

    plt.figure()
    plt.ylim(-1.2, 1.2)
    plt.title(f"wave eq. in 1D with {name}")
    plt.xlabel("x / a.u.")
    plt.ylabel(r"field, $\phi$ / a.u.")
    line, = plt.plot(xs, phi(0.0))

    def animate(i):
        line.set_ydata(phi(dt * i))

    return a.FuncAnimation(plt.gcf(), animate, interval=100, frames=nsteps)

def g(xi):
    return np.exp(-(xi - 0.7)**2*100) # gaussian

anim1 = plot_wave_eq_1d_solution("Dirichlet boundary condition",
    lambda xi: -g(-xi), g)
anim2 = plot_wave_eq_1d_solution("Neumann boundary condition",
    lambda xi: g(-xi), g)
plt.show()




