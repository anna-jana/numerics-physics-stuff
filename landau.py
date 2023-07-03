import numpy as np
import matplotlib.pyplot as plt

u = 1
r = 1
T_c = 1
m = np.linspace(-1, 1, 400)

# m ~ t^beta, beta = 1/2

def plot_f(T, label):
    t = (T - T_c) / T_c
    r0 = r * t
    f = r0 * m**2 + u * m**4
    plt.plot(m, f, label=label)

plt.figure()
plot_f(0.5 * T_c, "T < T_c")
plot_f(T_c, "T = T_c")
plot_f(1.5 * T_c, "T > T_c")
plt.xlabel("m")
plt.ylabel("f = F / N")
plt.title("Landau Theory f = r0 m^2 + u m^2, r0 = r t, t = (T - T_c) / T_c")
plt.legend()
plt.show()
