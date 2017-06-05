# Preisach simulation to model hysteresis curves
# * [1] F. Preisach, Über die magnetische Nachwirkung. Zeitschrift für Physik, 94:277-302, 1935
# Implemented after: June 2015, Markus Osterhoff

import numpy as np
import matplotlib.pyplot as plt

N = 32768 # number of hysterons
M = 2000 # number of steps
L = 2 # number of loops
r = 0.8 # abs max external field

# initialize the simulation
width = np.random.rand(N)
position = np.random.rand(N) - 0.5
alpha = position - width/2.
beta = position + width/2.
state = np.random.choice([-1., 1.], N)

xs = []; ys = []

def step(i):
    x = r*i/M
    global state, xs, ys
    state = np.where(x >= beta, 1.0, np.where(x <= alpha, -1.0, state))
    y = np.mean(state)
    xs.append(x)
    ys.append(y)
    # print "%+6.3f, %+6.3f" % (x, y)

# make L loops throu the ramping
for j in xrange(L):
    # ramp up
    for i in xrange(0, M + 1): step(i)
    # ramp down
    for i in xrange(M, -M - 1, -1): step(i)
    # ramp to 0
    for i in xrange(-M, 1, 1): step(i)

plt.plot(xs, ys)
plt.xlabel("Externes Feld")
plt.ylabel("Internes Feld")
plt.show()
