import numpy as np

def minimize(f, grad, x0, alpha=0.1, max_steps=200, eps=1e-10):
    x = x0
    for i in range(max_steps):
        new_x = x - alpha*grad(x)
        if np.linalg.norm(x - new_x) <= eps:
            return new_x
        x = new_x
    raise ValueError("failed to converge")


print minimize(lambda x: (x - 2)**2, lambda x: 2*x, 10.0)

print minimize(lambda x: x[0]**2 + x[1]**2,
               lambda x: np.array([2*x[0], 2*x[1]]),
               np.array([10.0, 10.9]))
