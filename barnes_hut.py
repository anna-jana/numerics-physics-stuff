import collections, functools, time, itertools
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

class BBox:
    def __init__(self, base, sizes):
        self.base = base
        self.sizes = sizes
        self.half_sizes = sizes / 2
        self.diameter = np.linalg.norm(sizes)

    def split(self, index):
        return BBox(self.base + index * self.half_sizes, self.half_sizes)

    @classmethod
    def from_xs(cls, xs):
        base = functools.reduce(np.minimum, xs)
        sizes = functools.reduce(np.maximum, xs) - base
        return cls(base, sizes)

class BHTree:
    def __init__(self, bbox, xs):
        dim = len(bbox.base.shape)
        self.bbox = bbox
        self.total_mass = len(xs)
        if len(xs) == 0:
            self.com = np.zeros(dim)
            self.subtrees = []
        elif len(xs) == 1:
            self.com = xs[0]
            self.subtrees = []
        else:
            subboxes = collections.defaultdict(lambda: [])
            for x in xs:
                subbox_index = tuple(np.round((x - bbox.base) / bbox.sizes))
                subboxes[subbox_index].append(x)
            self.subtrees = [BHTree(bbox.split(index), sub_xs) for index, sub_xs in subboxes.items()]
            self.com = sum(subtree.total_mass * subtree.com for subtree in self.subtrees) / self.total_mass

    def compute_force(self, x, eps, h, G):
        d = self.com - x
        dist = np.linalg.norm(d)
        if dist == 0.0:
            return np.zeros(len(x))
        if len(self.subtrees) <= 1 or self.bbox.diameter / dist < eps: # barnes hut condition
            print(len(self.subtrees) > 1)
            return G * self.total_mass / (h + dist)**3 * d # units/scales are in G
        else:
            return sum(subtree.compute_force(x, eps, h, G) for subtree in self.subtrees)

def compute_forces_bh(xs, eps, h, G):
    bbox = BBox.from_xs(xs)
    bhtree = BHTree(bbox, xs)
    return [bhtree.compute_force(x, eps, h, G) for x in xs]

def compute_force_direct(xs, h, G):
    forces = []
    for x in xs:
        force = np.zeros(len(xs[0]))
        for x_prime in x:
            d = x_prime - x
            dist = np.linalg.norm(d)
            if dist == 0.0:
                continue
            force += G / (h + dist)**3 * d
        forces.append(force)
    return forces

def run_velocity_verlet(compute_force, xs, vs, args, nsteps, dt):
    print("step: 0")
    Fs = compute_force(xs, *args)
    history = []
    for i in range(nsteps):
        print("step:", i + 1)
        xs = [x + dt*v + 0.5*dt**2*F for x, v, F in zip(xs, vs, Fs)]
        new_Fs = compute_force(xs, *args)
        vs = [v + dt*0.5*(F + new_F) for v, F, new_F in zip(vs, Fs, new_Fs)]
        Fs = new_Fs
        history.append(xs)
    return history

np.random.seed(42)

nstars = 1000
radius = 10.0
height = 0
G = 1e-3
rel_zv = 0.001
eps = 1.0
h = 1
tspan = 3
dt = 1

nsteps = int(tspan / dt)

alphas = np.random.uniform(0, 2*np.pi, nstars)
rs = np.random.uniform(0, radius, nstars)
np.sort(rs)
c, s = np.cos(alphas), np.sin(alphas)
x = rs * c
y = rs * s
z = np.random.uniform(-height/2, height/2, nstars)
xs = np.vstack([x, y, z]).T
M = np.arange(nstars) # integrated mass
v = 0 # np.sqrt(G * M / rs) # speed of circular orbit
vx = - v * s
vy =   v * c
vz = rel_zv * v * np.random.uniform(-0.5, 0.5, nstars)
vs = np.vstack([vx, vy, vz]).T

start = time.time()
history_bh = run_velocity_verlet(compute_forces_bh, xs, vs, (eps, h, G), nsteps, dt)
end = time.time()
print("bh time:", end - start)
start = time.time()
history_direct = run_velocity_verlet(compute_force_direct, xs, vs, (h, G), nsteps, dt)
end = time.time()
print("direct time:", end - start)

fig = plt.figure()
ax = fig.add_subplot() # projection="3d")
for i, c in zip(range(len(history_bh[0])), itertools.cycle(TABLEAU_COLORS)):
    ax.plot([history_bh[k][i][0] for k in range(len(history_bh))],
            [history_bh[k][i][1] for k in range(len(history_bh))], color=c)
            # [history_bh[k][i][2] for k in range(len(history_bh))])

for i, c in zip(range(len(history_direct[0])), itertools.cycle(TABLEAU_COLORS)):
    ax.plot([history_direct[k][i][0] for k in range(len(history_direct))],
            [history_direct[k][i][1] for k in range(len(history_direct))], ls="--", color=c)
            # [history_direct[k][i][2] for k in range(len(history_direct))])
plt.show()
