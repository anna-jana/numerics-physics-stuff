import numpy as np
import matplotlib.pyplot as plt

# based on: https://www.youtube.com/watch?v=ZUXmO4hu-20
# https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods

#################################### the mesh ###############################
# cartesian grid
# 9 discrete D2Q9 and macroscoptic velocities

# 6 2 5
# 3 0 1
# 7 4 8
n_discrete_velocities = 9

lattice_velocities = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
])

lattice_indicies = np.arange(n_discrete_velocities)

opposite_lattice_indicies = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

lattice_weights = np.array([ 4/9, # center
                             1/9, 1/9, 1/9, 1/9, # axis aligned
                             1/36, 1/36, 1/36, 1/36]) # diagonal velocities

right_velocities = np.array([1, 5, 8])
up_velocities = np.array([2, 5, 6])
left_velocities = np.array([3, 6, 7])
down_velocities = np.array([4, 7, 8])

pure_vertical_velocities = np.array([0, 2, 4])
pure_horizontal_velocities = np.array([0, 1, 3])

Nx = 150
Ny = 25

cylinder_r = Ny // 9
cylinder_x = Nx // 5
cylinder_y = Ny // 2

x = np.arange(Nx)
y = np.arange(Ny)
yy, xx = np.meshgrid(y, x)

cylinder_mask = (xx - cylinder_x)**2 + (yy - cylinder_y)**2 <= cylinder_r**2

###################################### flow parameters ################################
Re = 80.0
v_inflow = 0.04
velocity_profile = np.zeros((Nx, Ny, 2))
velocity_profile[:, :, 0] = v_inflow

nu = v_inflow * cylinder_r / Re
relaxation_factor = 1 / (3 * nu + 0.5) # omega

######################################### solver ##################################
def compute_density(v):
    return np.sum(v, axis=-1)

def compute_velocity(v, density):
    return sum(v[:, :, i, None] * lattice_velocities[None, None, i, :]
            for i in range(n_discrete_velocities)) / density[:, :, None]

def compute_equilibrium(density, velocity):
    projected_velocities = np.dstack([
        np.sum(lattice_velocities[None, None, i, :] * velocity[:, :, :], axis=-1)
        for i in range(n_discrete_velocities)])
    return density[:, :, None] * lattice_weights[None, None, :] * (
            1 +
            3*projected_velocities +
            9/2*projected_velocities**2 -
            3/2*(np.linalg.norm(velocity, axis=-1)**2)[:, :, None])

discrete_velocity = compute_equilibrium(np.ones((Nx, Ny)), velocity_profile)

nsteps = 14000
plot_every_nsteps = 100

def plot(velocity):
    dvx_dy = (np.roll(velocity[:, :, 0], -1, 1) - np.roll(velocity[:, :, 0], +1, 1)) / 2
    dvy_dx = (np.roll(velocity[:, :, 1], -1, 0) - np.roll(velocity[:, :, 1], +1, 0)) / 2
    vorticity = dvy_dx - dvx_dy
    plt.pcolormesh(x, y, vorticity.T, cmap="RdBu")
    plt.colorbar(orientation="horizontal", label="vorticity")
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.add_patch(plt.matplotlib.patches.Circle(
        (cylinder_x, cylinder_y), cylinder_r, color="black"))

for step in range(nsteps):
    print(f"{step = } / {nsteps}")
    # apply outflow bc on right bounary
    discrete_velocity[-1, :, left_velocities] = discrete_velocity[-2, :, left_velocities]

    # compute macroscopic quantitties
    density = compute_density(discrete_velocity)
    velocity = compute_velocity(discrete_velocity, density)

    # apply inflow profile by zou/he dirichelt bc
    velocity[0, 1:-1, :] = velocity_profile[0, 1:-1, :]
    # I have no idea why numpy does swap these axis and we need the T
    density[0, :] = (
            (compute_density(discrete_velocity[0, :, pure_vertical_velocities].T)
            + 2 * compute_density(discrete_velocity[0, :, left_velocities].T)) /
            (1 - velocity[0, :, 0])
        )

    # compute discrete eqilibrium velocities
    equilibrium = compute_equilibrium(density, velocity)

    # apply inflow profile by zou/he dirichelt bc (cont.)
    discrete_velocity[0, :, right_velocities] = equilibrium[0, :, right_velocities]

    # perform a collision step accourding to btk
    discrete_velocity_post_collision = discrete_velocity - relaxation_factor * (discrete_velocity - equilibrium)

    # apply bounce back bc on cylinder obstacle
    for i in range(n_discrete_velocities):
        discrete_velocity_post_collision[cylinder_mask, lattice_indicies[i]] = \
                discrete_velocity[cylinder_mask, opposite_lattice_indicies[i]]

    # steam alongsid the lattice velicities (discrete velocities are moved in their direction)
    discrete_velocity_streamed = discrete_velocity_post_collision
    for i in range(n_discrete_velocities):
        discrete_velocity_streamed[:, :, i] = \
                np.roll(discrete_velocity_streamed[:, :, i], lattice_velocities[i], (0, 1))

    discrete_velocity = discrete_velocity_streamed

plt.figure()
plot(velocity)
plt.show()
