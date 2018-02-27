import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.constants import gravitational_constant as G

# newtonian dynamics of an n body system
def rhs(y, t, *masses):
    num_planets = len(masses)
    y = y.reshape((num_planets, 2, 3))
    ans = y.copy()
    for i in range(num_planets):
        ans[i, 0, :] = y[i, 1, :] # dx/dt = v
        accel = np.zeros(3)
        for j in range(num_planets):
            if i != j:
                between = y[j, 0, :] - y[i, 0, :]
                dist = np.linalg.norm(between)
                accel += G*masses[j]/dist**3*between
        ans[i, 1, :] = accel # dv/dt = a
    return ans.reshape(ans.size)

# Parameter
# earth mars sun
masses = (5.97237e24, 6.4171e23, 1.988544e30) # kg

# Initial Conditions
# km, km/s
# Data from Ephemerides from some day
earth_pos = [-1.012268338703987E+08, -1.111875886682171E+08, -1.939665193599463E+04]
earth_vel = [2.152795356301499E+01, -2.018669837471565E+01,  1.000460883457954E-03]
mars_pos = [-1.345930796446981E+08, -1.863155816469951E+08, -6.188645620241463E+05]
mars_vel = [2.053477661794103E+01, -1.212126142710785E+01, -7.582624591585443E-01]
sun_pos = [5.626768185365887E+05,  3.432765388815567E+05, -2.436414149240617E+04]
sun_vel = [-7.612831360793502E-04,  1.210783982822092E-02, -1.274982680357986E-06]
y0 = np.array([
    [earth_pos, earth_vel],
    [mars_pos, mars_vel],
    [sun_pos, sun_vel],
]) # [km, km/s]
num_planets = y0.shape[0]
y0 *= 1000.0 # km -> m
y0 = y0.reshape(y0.size)

# Timeframe
T = 2*365*24*60*60.0 # 2y -> [s]
steps = 10000
ts = np.linspace(0, T, steps)

# Integration of the equations
ans = odeint(rhs, y0, ts, args=masses)
ans = ans.reshape((steps, num_planets, 2, 3))

# Plot the result
planet_names = ["Earth", "Mars", "Sun"]
colors = ["blue", "red", "yellow"]
for i in range(num_planets):
    xs = ans[:, i, 0, 0]/1000.0
    ys = ans[:, i, 0, 1]/1000.0
    plt.plot(xs, ys, label=planet_names[i], color=colors[i])
plt.legend(loc=3)
plt.title("Solarsystem Simulation")
plt.xlabel("x/km")
plt.ylabel("y/km")
plt.grid()
plt.show()

