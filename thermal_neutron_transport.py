# simulate the radiation transport of thermal neutrons in water using monte carlo integration

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

lambda_a = 45.0 # mean free path for absorption [cm^2]
lambda_s = 0.3 # mean free path for scattering [cm^2]

sigma_a = 1 / lambda_a
sigma_s = 1 / lambda_s
sigma_t = sigma_s + sigma_a

lambda_t = 1 / sigma_t

num_particles = 1000

x = np.zeros(num_particles)
y = np.zeros(num_particles)
z = np.zeros(num_particles)

absorped = np.zeros(num_particles, dtype=np.int)
history_length = np.zeros(num_particles)

while not np.all(absorped):
    s = - lambda_t * np.log(rand(num_particles))
    theta = np.arcsin(-1 + 2*rand(num_particles))
    phi = 2*np.pi*rand(num_particles)

    dx = s * np.cos(theta) * np.sin(phi)
    dy = s * np.cos(theta) * np.cos(phi)
    dz = s * np.sin(theta)

    x += dx
    y += dy
    z += dz

    absorped |= rand(num_particles) < sigma_a / sigma_t
    history_length += np.where(absorped, 0, 1)

r = np.sqrt(x**2 + y**2 + z**2)
print("<r> =", np.mean(r))
print("mean steps:", "analytical: lambda_a / lambda_s =", lambda_a / lambda_s, "numerical:", np.mean(history_length))

plt.hist(r)
plt.xlabel("r [cm]")
plt.ylabel("count out of %i" % (num_particles,))
plt.title("thermal neutron radiation transport in water")
plt.show()
