import random
import numpy as np
import matplotlib.pyplot as plt


# ## Polynomial Interpolation

def find_div_diffs(x_data, y_data):
    n = len(x_data)
    table = np.empty((n, n))
    table[:, 0] = y_data
    for length in range(1, n):
        for last_index in range(length, n):
            table[last_index, length] = (
                (table[last_index, length - 1] - table[last_index - 1, length - 1]) /
                (x_data[last_index] - x_data[last_index - length])
            )
    return np.diag(table)

def eval_newton_interp_poly(x_data, div_diffs, x):
    y = 0
    for i in range(len(x_data) - 1, 0, -1):
        y = (x - x_data[i - 1]) * (div_diffs[i] + y)
    y += div_diffs[0]
    return y

def poly_interp(x_data, y_data, x):
    div_diffs = find_div_diffs(x_data, y_data)
    return eval_newton_interp_poly(x_data, div_diffs, x)


# ## Rational Interpolation

def compute_inv_diff_quot(x, y):
    table = np.empty((y.size, y.size))
    table[:, 0] = y
    for col_idx in range(1, y.size):
        for row_idx in range(col_idx, y.size):
            table[row_idx, col_idx] = (
                (x[row_idx] - x[col_idx - 1]) /
                (table[row_idx, col_idx - 1] - table[col_idx - 1, col_idx - 1])
            )
    return np.diag(table)

def eval_rat_interp(x_data, x_val, inv_diff_quot):
    y_val = inv_diff_quot[-1]
    for i in range(x_data.size - 1, -1, -1):
        y_val = inv_diff_quot[i] + (x_val - x_data[i]) / y_val
    return y_val

def rat_interp(x_data, y_data, x):
    perm = np.arange(len(x_data))
    while True:
        random.shuffle(perm)
        x_data_p = x_data[perm]
        y_data_p = y_data[perm]
        inv_diff_quots = compute_inv_diff_quot(x_data_p, y_data_p)
        y_interp = eval_rat_interp(x_data_p, xs, inv_diff_quots)
        if np.all(np.isfinite(y_interp)):
            return y_interp

# ## Trigonometric Interpolation

def fft(a): # bad recursive version
    N = len(a)
    assert int(np.log2(N)) == np.log2(N)
    if len(a) == 1:
        return a
    a_hat = np.empty(len(a), dtype="complex")
    a_hat[::2] = fft(a[:N//2] + a[N//2:]) # even
    omega_N = np.exp(- 2*np.pi * 1j / N)
    j = np.arange(N//2)
    a_hat[1::2] = fft((a[:N//2] - a[N//2:]) * omega_N**j) # odd
    return a_hat

def fourier_approx(fn, M, n, L):
    assert n < M and M < L
    j = np.arange(M)
    x_samples = 2*np.pi*j/M
    f_samples = fn(x_samples)
    c_star = fft(f_samples) / M
    c_tilda = np.concatenate([c_star[:n+1], np.zeros(L - 2*n - 1), c_star[-n:]])
    f_vals = fft(c_tilda)
    return f_vals

def trig_interp(y_data, r):
    N = len(y_data)
    c_tilda = 1 / N * fft(y_data)
    ps = [y_data]
    ds = []
    for s in range(1, r - 1 + 1):
        rho = None
        d = c_tilda * rho
        p = fft(d)
        p[1:] = p[-1:0:-1]
        ps.append(p)
    ys = np.empty(r * N)
    for s in range(r):
        for k in range(N):
            ys[k*r + s] = ps[s][k]
    return ys

if __name__ == "__main__":
    f = lambda x: 1 / (1 + 25*x**2)

    x_data = np.linspace(-5, 5, 8 + 1)
    y_data = f(x_data)

    xs = np.linspace(np.min(x_data), np.max(x_data), 100)

    plt.plot(xs, f(xs), label="f")
    plt.plot(xs, rat_interp(x_data, y_data, xs), "--", label="rat interp")
    plt.plot(xs, poly_interp(x_data, y_data, xs), ":", label="poly interp")
    plt.plot(x_data, y_data, "o", label="interp. points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.ylim(-0.5, plt.ylim()[1])
    plt.show()

# ## Splines

# TODO




