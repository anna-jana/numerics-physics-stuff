import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.constants import gravitational_constant as G

plt.ion()
plt.style.use("ggplot")


# #y = n * 2 * 3
def make_rhs(masses):
    def rhs(y, t):
        n = y.size // (2 * 3)
        res = y.copy()
        for i in xrange(n):
            a = i * 2 * 3
            res[a:a+3] = y[a+3:a+2*3]
            accel = np.zeros(3)
            for j in xrange(n):
                b = j * 2 * 3
                between = y[b:b+3] - y[a:a+3]
                d = np.linalg.norm(between)
                if d == 0.0: continue
                accel += G*masses[j]/d**3*between
            res[a+3:a+6] = accel
        return res
    return rhs

# earth mars
masses = np.array([5.97237e24, 6.4171e23, 1.988544e30]) # kg

# km, km/s
earth_pos = [-1.012268338703987E+08, -1.111875886682171E+08, -1.939665193599463E+04]
earth_vel = [2.152795356301499E+01, -2.018669837471565E+01,  1.000460883457954E-03]
mars_pos = [-1.345930796446981E+08, -1.863155816469951E+08, -6.188645620241463E+05]
mars_vel = [2.053477661794103E+01, -1.212126142710785E+01, -7.582624591585443E-01]
sun_pos = [5.626768185365887E+05,  3.432765388815567E+05, -2.436414149240617E+04]
sun_vel = [-7.612831360793502E-04,  1.210783982822092E-02, -1.274982680357986E-06]


# n: number of bodies
# 2: x, v
# 3: x, y, z for x and vx, vy, vz for v
# n * 2 * 3

rhs = make_rhs(masses)

data = [
    [earth_pos, earth_vel],
    [mars_pos, mars_vel],
    [sun_pos, sun_vel],
]
y0 = np.array(data) # km, km/s

y0 *= 1000.0 # km -> m

y0 = y0.reshape(len(data)*2*3)

T = 2*365*24*60*60.0 # 2y -> [s]
steps = 10000
ts = np.linspace(0, T, steps)

res = odeint(rhs, y0, ts)

for i in range(res.shape[1] // (2 * 3)):
    x = res[:, i*2*3]
    y = res[:, i*2*3 + 1]
    plt.plot(x, y)

raw_input()
