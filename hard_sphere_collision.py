import numpy as np
import matplotlib.pyplot as plt

r1 = 1.0
r2 = 1.0

m1 = 1.0
m2 = 1.0

x1 = np.array([0, 0])
x2 = np.array([0, r1 + r2])

v1 = np.array([-0.5, 0.25])
v2 = np.array([-1.5, -1.2])

# displacement vector for the two hard spheres
d = x1 - x2
d /= np.linalg.norm(d)

# parallel to d
v1_parallel = np.dot(d, v1)
v2_parallel = np.dot(d, v2)

# orthogonal to d
v1_orthogonal = v1 - v1_parallel * d
v2_orthogonal = v2 - v2_parallel * d

P = m1*v1_parallel + m2*v2_parallel # total momentum
E = 0.5*m1*v1_parallel**2 + 0.5*m2*v2_parallel**2 # total energy

# solve the 1d collision along d using energy and momentum conservation
a = m2**2/m1 + m2
b = -2*m2/m1*P
c = P**2/m1 - 2*E
p = b/a
q = c/a
v2_prime_parallel = -(p/2)**2 + np.sqrt((p/2)**2 - q)
v1_prime_parallel = (P - m2*v2_prime_parallel)/m1

v1_prime = v1_orthogonal + v1_prime_parallel * d
v2_prime = v2_orthogonal + v2_prime_parallel * d

# plot
fig, ax = plt.subplots(1, 1)
ax.set_xlim(-1 + min(x1[0] - r1, x2[0] - r2), 1 + max(x1[0] + r1, x2[0] + r2))
ax.set_ylim(-1 + min(x1[1] - r1, x2[1] - r2), 1 + max(x1[1] + r1, x2[1] + r2))
ax.add_artist(plt.Circle(x1, r1))
ax.add_artist(plt.Circle(x2, r2))
ax.annotate("", x1, x1 + v1, arrowprops=dict(arrowstyle="<-"))
ax.annotate("", x2, x2 + v2, arrowprops=dict(arrowstyle="<-"))
ax.annotate("", x1, x1 + v1_prime, arrowprops=dict(color="red", arrowstyle="<-"))
ax.annotate("", x2, x2 + v2_prime, arrowprops=dict(color="red", arrowstyle="<-"))
ax.set_aspect("equal")
plt.show()



