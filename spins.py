import numpy as np, matplotlib.pyplot as plt
import scipy.linalg as la, scipy.sparse as sp
import functools

# I call S_x the up/down spins - sorry

pauli_matricies = 0.5 * np.array([
    [[0, 1],
     [1, 0]],
    [[0, -1j],
     [1j, 0]],
    [[1, 0],
     [0, -1]],
])

I = np.eye(2)

spin_up_down_basis = np.array([
    [1, 1],
    [-1, 1],
]) / np.sqrt(2)

def make_many_body_op(single_body_op, side, nsides):
    ops = [single_body_op if i == side else I for i in range(nsides)]
    return functools.reduce(sp.kron, ops)

def make_interaction_between(i, j, nsides):
    return sum(make_many_body_op(pauli_matricies[k], i, nsides) @
               make_many_body_op(pauli_matricies[k], j, nsides)
             for k in range(3))

def make_heisenberg_hamiltonian(nsides, coupling):
    return - coupling * sum(
            make_interaction_between(i, (i + 1) % nsides, nsides)
            for i in range(nsides))

def make_up_down_state(spin_config):
    return functools.reduce(np.kron, (spin_up_down_basis[spin] for spin in spin_config))

def make_neel_config(nsides):
    return np.arange(nsides) % 2

def make_single_spin_config(side, nsides):
    return (side == np.arange(nsides)).astype("int")

def compute_expectation_value(op, state):
    return np.real(state.T.conj() @ op @ state)

def compute_up_down_expectation_value(state):
    nstates = int(np.log2(len(state)))
    Sx = pauli_matricies[0]
    return [compute_expectation_value(make_many_body_op(Sx, i, nstates), state)
            for i in range(nstates)]

def compute_time_evolution(H, init, ts):
    # TODO: use only a few dominant eigenvectors and the sparse routine
    # or own laczos procedure
    Es, Psis = la.eigh(H.toarray())
    psi0 = Psis.T.conj() @ init
    up_down_basis_evolution = [Psis @ (np.exp(- 1j * Es * t) * psi0) for t in ts]
    return up_down_basis_evolution

def sim(spin_config, coupling, tspan, nsteps=100):
    H = make_heisenberg_hamiltonian(len(spin_config), coupling)
    init = make_up_down_state(spin_config)
    ts = np.linspace(0, tspan, nsteps)
    up_down_basis_evolution = compute_time_evolution(H, init, ts)
    up_down_expectation_values = np.array([compute_up_down_expectation_value(s)
        for s in up_down_basis_evolution])
    plt.figure()
    plt.pcolormesh(np.arange(up_down_expectation_values.shape[1]),
            ts,
            up_down_expectation_values)
    plt.colorbar(label="expectation value of S_x (up/down spin)")
    plt.xlabel("sides")
    plt.ylabel("time")
    plt.show()

# sim(make_neel_config(6), 1.0, 10.0)
sim(make_single_spin_config(2, 5), 1.0, 10.0)
