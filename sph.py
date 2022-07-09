import numpy as np, matplotlib.pyplot as plt, sympy as sp
np.random.seed(42)

################################# initialize system #############################
########## physical parameters #########
# TODO: choose reasonable values
g = 1
k = 1
rho_0 = 1
width = 1.0
height = 3.0
v0 = 1e-1
Re = 10
fill_height_percent = 0.7
reflection_damping = 0.75

def is_inside(x, y):
    return 0 <= x <= width and 0 <= y <= height

######### particles ##########
#### water in a cup ####
nparticles = 50
xs = np.random.uniform(0, width, nparticles)
ys = np.random.uniform(0, fill_height_percent * height, nparticles)
rs = np.vstack([xs, ys]).T
v = np.random.uniform(0, v0, nparticles)
angle = np.random.uniform(0, 2*np.pi, nparticles)
vxs = v * np.cos(angle)
vys = v * np.sin(angle)
vs = np.vstack([vxs, vys]).T

m = rho_0 * width * height / nparticles

############ boxes ##########
h = 0.3
nboxes_x = int(np.ceil(width / h))
nboxes_y = int(np.ceil(height / h))

box_counts = np.zeros((nboxes_x, nboxes_y), dtype="int")
box_particle_ids = np.zeros((nboxes_x, nboxes_y, nparticles), dtype="int")

def get_box(x, y):
    return int(x / h), int(y / h)

for i in range(nparticles):
    index = get_box(xs[i], ys[i])
    box_particle_ids[index + (box_counts[index],)] = i
    box_counts[index] += 1

def neighboring_particles(x, y, xs, ys):
    n, m = get_box(x, y) # box in which the particle i is contained
    for dn in (-1, 0, 1): # iterate over all neighboring (including the central) boxes
        for dm in (-1, 0, 1):
            box = n + dn, m + dm
            if 0 <= box[0] < nboxes_x and 0 <= box[1] < nboxes_y: # ignore boxes outside the simultation volume
                for box_index in range(box_counts[box]): # go over the particles in the box
                    j = box_particle_ids[box + (box_index,)]
                    if not is_inside(xs[j], ys[j]):
                        continue
                    yield j

########### time integration ##########
tspan = 2.0
dt = 1e-3
nsteps = int(np.ceil(tspan / dt))

##################################### kernels ###################################
def make_spiky_grad():
    h, x, y = sp.symbols("h, x, y")
    r = sp.sqrt(x**2 + y**2)
    expr = (h - r)**3
    grad = [expr.diff(x), expr.diff(y)]
    print(grad)

C_poly6 = 315 / (64*np.pi*h**9)

def calc_kernel_poly6(r):
    return C_poly6 * np.where(r <= h, (h**2 - r**2)**3, 0.0)

C_spiky = 15 / (np.pi*h**6)

null = np.zeros(2)

def calc_grad_spiky(x, y):
    r = np.sqrt(x**2 + y**2)
    if r > h:
        return null
    C = C_spiky * (-3) * (h - r)**2/r
    return C * np.array([x, y])

C_viscosity = 15 / (2*np.pi*h**3)

def make_viscosity_laplacian():
    h, x, y, r = sp.symbols("h, x, y, r")
    r = sp.sqrt(x**2 + y**2)
    expr = -r**3/(2*h**3) + r**2/h**2 + h / (2*r) - 1
    laplacian = expr.diff(x, 2) + expr.diff(y, 2)
    print(laplacian)

def calc_viscosity_laplacian(x, y):
    r = np.sqrt(x**2 + y**2)
    if r > h:
        return 0.0
    return C_viscosity * (0.5*h/r**3 + 4/h**2 - 9/2*r/h**3)

######################################## force computation #################################
f_gravity = g * np.array([0, -1]) # this is constant

def calc_pressure(rho):
    return k * (rho - rho_0)

# calculate all densities at the particle locations
def compute_densities(xs, ys):
    def compute_density(x, y):
        return sum(calc_kernel_poly6(np.sqrt((x - xs[j])**2 + (y - ys[j])**2)) * m
                for j in neighboring_particles(x, y, xs, ys))
    return [compute_density(xs[i], ys[i]) for i in range(xs.shape[0])]

def compute_forces(rs, vs):
    xs, ys = rs[:,0], rs[:,1]
    rhos = compute_densities(xs, ys)
    # compute force for each particle
    force = np.zeros((rs.shape[0], 2)) + f_gravity
    # calculate forces
    for i in range(rs.shape[0]):
        p_i = calc_pressure(rhos[i])
        for j in neighboring_particles(xs[i], ys[i], xs, ys):
            if i == j:
                continue
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            p_j = calc_pressure(rhos[j])
            grad_W_pressure = calc_grad_spiky(dx, dy)
            laplace_W_viscosity = calc_viscosity_laplacian(dx, dy)
            f_pressure = - m * (p_i + p_j) / (2*rhos[j]) * grad_W_pressure
            f_viscosity = 1 / Re * m * (vs[j] - vs[i]) / rhos[j] * laplace_W_viscosity
            force[i] += f_pressure + f_viscosity
    return force

eps = 1e-8

def reflect_on_boundary(coord_normal, coord_tanj, vel_normal, vel_tanj, normal, boundary):
    diff = coord_normal - boundary
    if normal * diff < 0.0:
        t_collision = diff / vel_normal # positive

        new_vel_normal = - reflection_damping * vel_normal
        new_vel_tanj = reflection_damping * vel_tanj

        new_coord_normal = boundary + t_collision * new_vel_normal
        new_coord_tanj = coord_tanj - t_collision * vel_tanj + t_collision * new_vel_tanj

        return new_coord_normal, new_coord_tanj, new_vel_normal, new_vel_tanj
    return coord_normal, coord_tanj, vel_normal, vel_tanj

def apply_reflective_boundary_conditions(rs, vs):
    for i in range(rs.shape[0]):
        x, y, vx, vy = rs[i, 0], rs[i, 1], vs[i, 0], vs[i, 1]
        if vx == 0.0 and vy == 0.0: continue
        x, y, vx, vy = reflect_on_boundary(x, y, vx, vy, +1, 0)
        x, y, vx, vy = reflect_on_boundary(x, y, vx, vy, -1, width)
        y, x, vy, vx = reflect_on_boundary(y, x, vy, vx, -1, height)
        y, x, vy, vx = reflect_on_boundary(y, x, vy, vx, +1, 0)
        rs[i, 0], rs[i, 1], vs[i, 0], vs[i, 1] = x, y, vx, vy

############################### integration ##############################
# velocity verlet integration
do_plot = True
if do_plot:
    plt.figure()
Fs = compute_forces(rs, vs)
for step in range(nsteps):
    print("step:", step + 1, "of", nsteps, "@ t =", dt*step)
    # euler:
    # Fs = compute_forces(rs, vs)
    # rs += dt*vs
    # vs += dt*Fs
    # verlet:
    rs += dt*vs + 0.5*dt**2*Fs
    new_Fs = compute_forces(rs, vs + dt*Fs) # I am not sure if this is allowed
    vs += 0.5*dt*(Fs + new_Fs)
    Fs = new_Fs
    apply_reflective_boundary_conditions(rs, vs)

    if do_plot:
        plt.clf()
        xs, ys, vxs, vys = rs[:,0], rs[:,1], vs[:,0], vs[:,1]
        for i in range(nboxes_x):
            for j in range(nboxes_y):
                x = i*h
                y = j*h
                plt.plot([x, x, x + h, x + h, x], [y, y + h, y + h, y, y],
                        "r", lw=0.5)
        plt.plot(xs, ys, "o")
        plt.quiver(xs, ys, vxs, vys)
        plt.plot([0, 0, width, width, 0], [height, 0, 0, height, height], "k", lw=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.0001)

######### plotting ###########
plt.show()

