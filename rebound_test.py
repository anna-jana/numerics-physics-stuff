import numpy as np, matplotlib.pyplot as plt
import rebound
import tqdm

plt.ion()
np.random.seed(42)

# milkyway:
def make(
        # number of test particles (not starts that would be 100-400 10^9)
        nparticles = 10,
        R = 100, # [AU]
        M = 1, # M_solar
    ):
    m = M / nparticles # M_solar
    r = np.random.uniform(0, R, nparticles - 1)
    omega = np.random.uniform(0, 2*np.pi, nparticles - 1)
    r.sort()

    sim = rebound.Simulation()
    sim.units = ("AU", "Yr", "Msun")
    sim.add(m = m)
    for i in range(nparticles - 1):
        sim.add(m = m, a = r[i], omega = omega[i],)
    sim.move_to_com()

    return sim

def run(
        sim,
        tmax = 5e2,
        steps = 100,
    ):
    ts = np.linspace(0, tmax, steps)
    pos = np.empty((steps, len(sim.particles), 2))
    for i, t in tqdm.tqdm(list(enumerate(ts))):
       sim.integrate(t)
       for j, p in enumerate(sim.particles):
           pos[i, j, 0] = p.x
           pos[i, j, 1] = p.y
    return pos

def plot(pos):
    plt.figure()
    nsteps, nparticles, _ = pos.shape
    for j in range(nparticles):
        l, = plt.plot(pos[:, j, 0], pos[:, j, 1]) # , color="black")
        plt.plot(pos[0, j, 0], pos[0, j, 1], "o", color=l.get_color())
    plt.xlabel("x / AU")
    plt.ylabel("y / AU")

def main():
    sim = make()
    pos = run(sim)
    plot(pos)
