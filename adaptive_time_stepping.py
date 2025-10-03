import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, y, t, h):
    k1 = f(t,       y)
    k2 = f(t + h/2, y + h/2*k1)
    k3 = f(t + h/2, y + h/2*k2)
    k4 = f(t + h,   y + h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

def solve_ode(f, t_start, t_end, y, atol, rtol, stiffness=0.5):
    dt = (t_end - t_start) / 2.0
    ts = []
    ys = []
    t = t_start
    order = 5

    while True:
        # ans = single + A*(dt)**order
        # ans = two_half + A*(dt/2)**order
        # substract: two_half - single = dt**order * A * (1 - (1/2)**order)
        # (two_half - single) / (1 - (1/2)**order) = A * dt**order = err
        y_half_step = rk4_step(f, y, t, dt / 2.0)
        y_two_half_steps = rk4_step(f, y_half_step, t + dt/2, dt / 2.0)
        y_single_step = rk4_step(f, y, t, dt)

        error = np.linalg.norm((y_two_half_steps - y_single_step) / (1 - 0.5**order))

        scale = np.linalg.norm(rtol * np.abs(y) + atol)

        if not np.isclose(error, 0.0):
            dt *= stiffness * (scale / error)**(1/order)

        if error < scale:
            y = rk4_step(f, y, t, dt)
            t += dt
            end = t > t_end
            ys.append(y)
            ts.append(t)
            if end:
                break

    return np.array(ts), np.array(ys)

if __name__ == "__main__":
    atol = rtol = 1e-5
    ts, ys = solve_ode(lambda t, y: np.array([y[1], -y[0]]),
                   0.0, 10.0, [1.0, 0.0], atol, rtol)

    plt.figure(layout="constrained")
    plt.subplot(2,1,1)
    plt.plot(ts, ys[:, 0], ".", label="adaptive rk4")
    ts_ana = np.linspace(ts[0], ts[-1], 100)
    plt.plot(ts_ana, np.cos(ts_ana), label="analytic")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.subplot(2,1,2)
    plt.plot(ts, np.abs(np.cos(ts) - ys[:, 0]) / atol)
    plt.ylabel("actual absolute error /\ngoal absolute error")
    plt.xlabel("t")
    plt.show()
