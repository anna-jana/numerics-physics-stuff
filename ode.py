import numpy as np
import matplotlib.pyplot as plt

################################ methods ###################################
def forward_euler(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    h = t[1] - t[0]
    for i in range(t.size - 1):
        ys[i + 1, :] = ys[i,:] + h*f(t[i], ys[i,:])
    return ys

def heun(f, y0, t): # also called rk2
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    h = t[1] - t[0]
    for i in range(t.size - 1):
        k1 = f(t[i], ys[i,:])
        k2 = f(t[i] + h/2, ys[i,:] + h/2*k1)
        ys[i + 1, :] = ys[i,:] + h*(k1 + k2)/2
    return ys

def ab2(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    h = t[1] - t[0]
    ys[1, :] = forward_euler(f, y0, t[0:2])[1, :]
    for i in range(1, len(t) - 1):
        ys[i + 1, :] = ys[i, :] + 1.5 * f(t[i], ys[i, :]) * h - 0.5 * f(t[i - 1], ys[i - 1, :]) * h
    return ys

def ab3(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    ys[1, :] = forward_euler(f, y0, t[0:2])[1, :]
    ys[2, :] = ab2(f, ys[1, :], t[1:3])[1, :]
    h = t[1] - t[0]
    for i in range(2, len(t) - 1):
        ys[i + 1, :] = ys[i, :] + h / 12.0 * (23 * f(t[i], ys[i, :]) - 16 * f(t[i - 1], ys[i - 1, :]) + 5*f(t[i - 2], ys[i - 2]))
    return ys

def rk4(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0,:] = y0
    h = t[1] - t[0]
    for i in range(t.size - 1):
        k1 = f(t[i],       ys[i,:])
        k2 = f(t[i] + h/2, ys[i,:] + h/2*k1)
        k3 = f(t[i] + h/2, ys[i,:] + h/2*k2)
        k4 = f(t[i] + h,   ys[i,:] + h*k3)
        ys[i+1,:] = ys[i,:] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return ys

def leap_frog(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    h = t[1] - t[0]
    ys[1, :] = forward_euler(f, y0, t[0:2])[1,:]
    for i in range(1, len(t) - 1):
        ys[i + 1, :] = ys[i - 1, :] + 2.0 * f(t[i], ys[i, :])*h
    return ys

def dormand_prince_step(f, t, y, dt):
    k1 = f(t,           y)
    k2 = f(t + dt*1/5 , y + dt*(k1*1/5))
    k3 = f(t + dt*3/10, y + dt*(k1*3/40        + k2*9/40))
    k4 = f(t + dt*4/5 , y + dt*(k1*44/45       + k2*-56/15      + k3*32/9))
    k5 = f(t + dt*8/9 , y + dt*(k1*19372/6561  + k2*-25360/2187 + k3*64448/6561  + k4*-212/729))
    k6 = f(t + dt     , y + dt*(k1*9017/3168   + k2*-355/33     + k3*46732/5247  + k4*49/176    + k5*-5103/18656))
    return y + dt*(k1*35/384 + k3*500/1113 + k4*125/192 + k5*-2187/6784 + k6*11/84)

def dormand_prince(f, y0, t):
    ys = np.zeros((t.size, y0.size))
    ys[0, :] = y0
    for i in range(len(t) - 1):
        ys[i + 1, :] = dormand_prince_step(f, t[i], ys[i, :], t[i + 1] - t[i])
    return ys

################### test cases #####################

## harmonic oscillator

def test_harmonic_oscillator():
    def exact_harm_osc(t,x0,v0,m,k):
        c = k/m
        B = np.sqrt(c)
        C = np.arctan(-v0/(x0*B))
        A = x0/np.cos(C)
        return A*np.cos(B*t + C)

    T = 20
    steps = 1000
    t = np.linspace(0, T, steps)
    k = 2.3
    m = 1.2
    c = k/m
    x0 = 100.0
    v0 = 1.2
    y0 = np.array([x0, v0])
    harm_osc_rhs = lambda t, y: np.array([y[1], -c*y[0]])
    exact_xs = exact_harm_osc(t, x0, v0, m, k)

    def test_harm(name, integrator):
        ys = integrator(harm_osc_rhs, y0, t)
        xs = ys[:, 0]
        vs = ys[:, 1]
        err = np.abs(xs - exact_xs)
        plt.subplot(2, 1, 1)
        plt.plot(t, xs, label=name)
        plt.subplot(2, 1, 2)
        plt.semilogy(t, err, label=name)

    test_harm("forward_euler", forward_euler)
    test_harm("heun (rk2)", heun)
    test_harm("rk4", rk4)
    test_harm("leap_frog", leap_frog)
    test_harm("AB 2", ab2)
    test_harm("AB 3", ab3)
    test_harm("dormand prince", dormand_prince)

    plt.subplot(2, 1, 1)
    plt.plot(t, exact_xs, "--", label="analytic")
    plt.legend()
    plt.grid()
    plt.title(r"$x'' = -\frac{k}{m}x$")
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(2, 1, 2)
    plt.legend()
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("absolute Error")

## radioactive decay y' = - alpha * y

def test_radioactive_decay():
    alpha = 0.7
    x0 = 100.0

    y0 = np.array([x0])

    t = np.linspace(0, 10.0, 200)

    def radioactive_decay_rhs(t, y):
        return np.array([- alpha * y[0]])

    exact_xs = x0 * np.exp(- alpha * t)

    def test_radio(name, integrator):
        ys = integrator(radioactive_decay_rhs, y0, t)
        xs = ys[:, 0]
        err = np.abs(xs - exact_xs)
        plt.subplot(2, 1, 1)
        plt.plot(t, xs, label=name)
        plt.subplot(2, 1, 2)
        plt.semilogy(t, err, label=name)

    test_radio("forward_euler", forward_euler)
    test_radio("heun (rk2)", heun)
    test_radio("rk4", rk4)
    test_radio("leap_frog", leap_frog)
    test_radio("AB 2", ab2)
    test_radio("AB 3", ab3)
    test_radio("dormand prince", dormand_prince)

    plt.subplot(2, 1, 1)
    plt.plot(t, exact_xs, "--", label="analytic solution")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(r"$x' = - \alpha x$")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.xlabel("t")
    plt.ylabel("absolute error")
    plt.grid()
    plt.legend()

if __name__ == "__main__":
    plt.figure()
    test_harmonic_oscillator()
    plt.figure()
    test_radioactive_decay()
    plt.show()

