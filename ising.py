import numpy as np
import matplotlib.pyplot as plt
import tqdm
import numba
import time

@numba.jit(nopython=True)
def calc_Delta_H(spins, spin_coupling, row, col):
    rows, cols = spins.shape
    return (2 * spin_coupling * spins[row, col] *
            (spins[(row + 1) % rows, col] + spins[(row - 1) % rows, col] +
             spins[row, (col + 1) % cols] + spins[row, (col - 1) % cols]))

@numba.jit(nopython=True)
def mc_step(spins, spin_coupling, magnetization, energy):
    rows, cols = spins.shape
    for i in range(spins.size):
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)
        Delta_H = calc_Delta_H(spins, spin_coupling, row, col)
        p = min((1, np.exp(- Delta_H)))
        if np.random.rand() <= p:
            spins[row, col] *= -1
            energy -= Delta_H
            magnetization += 2*spins[row, col]
    return magnetization, energy

class Ising2D:
    """
    Ising model without external field. Rescaled spin coupling J -> J / beta.
    2D rectangluar grid, horizontal and vertical couplings to direct neighbors.
    Metropolis MCMC simulation.
    """
    def __init__(self, rows, cols, spin_coupling):
        self.rows, self.cols = rows, cols
        self.spins = 2*np.random.randint(0, 2, (rows, cols)) - 1
        self.spin_coupling = spin_coupling
        self.magnetization = self.compute_magnetization()
        self.energy = self.compute_hamiltonian()

    def compute_hamiltonian(self):
        return - self.spin_coupling * np.sum(sum(self.spins * np.roll(self.spins, (dn, dm))
                for dn, dm in [(1,0), (0,1), (-1,0), (0,-1)])) / 2

    def compute_magnetization(self):
        return np.sum(self.spins)

    def mc_step(self):
        self.magnetization, self.energy = mc_step(self.spins, self.spin_coupling, self.magnetization, self.energy)

def do_ising_sim(J):
    mc_steps = 100000
    ising = Ising2D(16, 16, J)
    ms = []
    start = time.time()
    for i in tqdm.tqdm(range(mc_steps)):
        ising.mc_step()
        ms.append(ising.magnetization)
    stop = time.time()
    print("simulation took:", stop - start, "seconds")
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(ms)
    ax1.set_title(f"Ising Model at J = {J}")
    ax1.set_xlabel("MC step")
    ax1.set_ylabel("Magnetization")
    ax2.hist(ms, histtype="step", density=True)
    ax2.set_xlabel("Magnetization")
    ax2.set_ylabel("Probability")
    plt.tight_layout()

J_c = 0.5*np.log(1+np.sqrt(2))
J_para = 0.4
J_ferro = 0.5

if __name__ == "__main__":
    do_ising_sim(J_para)
    do_ising_sim(J_c)
    do_ising_sim(J_ferro)
    plt.show()
