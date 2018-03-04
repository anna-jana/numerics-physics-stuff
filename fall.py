import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.integrate as inte

plt.style.use("ggplot")

# y = [x, v]
# y' = [v, a]

G = 6.67e-11 # m^3/s^2/kg
R = 149.6e9 # m
r = 6.9e8 # m
M = 1.99e30 # kg

def rhs(y, t):
    return y[1], -G*M/y[0]**2

y0 = np.array([R, 0.0])
t = 0.6e7 # s
steps = 1000
ts = np.linspace(0.0, t, steps)
xs = inte.odeint(rhs, y0, ts)[:, 0]

for i, x in enumerate(xs):
    if x <= r:
        T = ts[i] # s
        break

weeks = T / (7 * 24 * 60 * 60)
my_label = "Einschlag nach " + str(weeks) + " wochen"
plt.plot(ts, np.repeat(r, steps), label="Erde")
plt.scatter([T], [r], label=my_label)
plt.plot(ts, xs, label="Sonnen Oberflaeche")
plt.title("Die Erde faellt auf die Sonne")
plt.xlabel("t/s")
plt.ylabel("x/m")
plt.legend()
