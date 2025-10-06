import os.path
import os
import numpy as np
import tqdm
# from numba import njit # TODO

############################ SU(3) group #########################
def adjoint(A):
    return A.T.conj()

def reproject_su3(A):
    A[:, 0] /= np.linalg.norm(A[:, 0])
    A[:, 1] -= A[:, 0] * np.dot(A[:, 0], A[:, 1])
    A[:, 1] /= np.linalg.norm(A[:, 1])
    A[:, 2] = np.cross(A[:, 0], A[:, 1])

I = np.eye(2, dtype=np.complex128)

pauli = np.array([[[0, 1],
                   [1, 0]],
                  [[0, -1j],
                   [1j, 0]],
                  [[1, 0],
                   [0, -1]]], dtype=np.complex128)

def random_su2():
    r = np.random.rand(4) - 1.0
    eps = 1e-3
    x = eps * r / np.linalg.norm(r)
    sign = -1 if np.random.randint(2) == 0 else +1
    x0 = sign * np.sqrt(1 - eps**2)
    return (
        x0 * I +
        1j * x[0] * pauli[0] +
        1j * x[1] * pauli[1] +
        1j * x[2] * pauli[2]
    )

def random_su3():
    r = random_su2()
    s = random_su2()
    t = random_su2()
    R = np.array([[r[0, 0], r[0, 1], 0      ],
                  [r[1, 0], r[1, 1], 0      ],
                  [0,       0,       1      ]])
    S = np.array([[s[0, 0], 0,       s[0, 1]],
                  [0,       1,       0      ],
                  [s[1, 0], 0,       s[1, 1]]])
    T = np.array([[1,       0,       0      ],
                  [0,       t[0, 0], t[0, 1]],
                  [0,       t[1, 0], t[1, 1]]])
    X = R @ S @ T
    reproject_su3(X)
    return X

##################################### lattice operations ##########################
def get_link(field, i, j, k, l, d, direction):
    N = field.shape[0]
    Nt = field.shape[3]
    # link in -1 direction on axis d from node {A_n}
    #                   is the same as
    # link in +1 direction on axis d from neighboring node in -1 dieection on axis d
    if direction == -1:
        if d == 0:
            i -= 1
        elif d == 1:
            j -= 1
        elif d == 2:
            k -= 1
        else:
            l -= 1
    i %= N
    j %= N
    k %= N
    l %= Nt
    return field[i, j, k, l, d, :, :]

def get_link_with_offset(field, i, j, k, l, d_offset, offset_direction, d, direction):
    # add offset to index
    if d_offset == 0:
        i += offset_direction
    elif d_offset == 1:
        j += offset_direction
    elif d_offset == 2:
        k += offset_direction
    else:
        l += offset_direction
    return get_link(field, i, j, k, l, d, direction)

def stapel_of_link(field, i, j, k, l, d, d_offset, offset_direction):
    #                    ^              U2
    #                    |          +-----+
    # d_offset, diection |   U1 --> |     | <-- U3
    #                    | (i,j,k,l)+=====+
    #                    |             ^
    #                    |         changed link
    #                    +--------------------->
    #                        d, always in +1 direction
    U1 = get_link(field, i, j, k, l, d_offset, offset_direction)
    U2 = get_link_with_offset(field, i, j, k, l, d_offset, offset_direction, d, 1)
    U3 = get_link_with_offset(field, i, j, k, l, d, 1, d_offset, offset_direction)
    return U1 @ U2 @ U3

######################################### physics ######################################
def compute_action_diff(beta, field, i, j, k, l, d, old, new):
    A = np.zeros((3, 3), dtype=np.complex128)
    for d_offset in range(4):
        if d_offset != d:
            for direction in (-1, +1):
                A += stapel_of_link(field, i, j, k, l, d, d_offset, direction)
    N = field.shape[0]
    return - beta / N * np.real(np.trace((new - field[i, j, k, l, d, direction]) @ A))

def compute_polyakov_loop(field, P):
    N = field.shape[0]
    Nt = field.shape[3]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                us = np.eye(3, dtype=np.complex128)
                for l in range(Nt):
                    us @= get_link(field, i, j, k, l, 3, 1)
                P[i, j, k] = np.trace(us)

################################### algorithm ##############################
def run_simulation(beta, field, nsteps):
    N = field.shape[0]
    Nt = field.shape[3]
    reproject_every = 10
    P = np.empty((N, N, N), dtype=np.complex128)
    for mc_step in tqdm.tqdm(range(nsteps)):
        # mc sweep
        for _ in range(N**3 * Nt):
            # metropolis
            i = np.random.randint(N)
            j = np.random.randint(N)
            k = np.random.randint(N)
            l = np.random.randint(Nt)
            d = np.random.randint(4)
            old = field[i, j, k, l, d, :, :]
            new = random_su3()
            action_diff = compute_action_diff(beta, field, i, j, k, l, d, old, new)
            if np.exp(-action_diff) <= np.random.rand():
                field[i, j, k, l, d, :, :] = new
        # reproject
        if mc_step % reproject_every == 0:
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        for l in range(Nt):
                            for d in range(4):
                                reproject_su3(field[i, j, k, l, d, :, :])
        # observables
        compute_polyakov_loop(field, P)

def load(prefix):
    name = os.path.join(prefix, "field.npy")
    return np.load(name)

def save(prefix, field):
    np.save(os.path.join(prefix, "field"), field)

def start(prefix, N, Nt, beta):
    os.mkdir(prefix)
    with open(os.path.join(prefix, "beta.txt"), "w") as fh:
        fh.write(str(beta))
    field = np.empty((N, N, N, Nt, 4, 3, 3), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(Nt):
                    for d in range(4):
                        field[i, j, k, l, d, :, :] = random_su3()
    save(prefix, field)

def run(prefix, nsteps):
    field = load(prefix)
    with open(os.path.join(prefix, "beta.txt"), "r") as fh:
        beta = float(fh.read())
    run_simulation(beta, field, nsteps)
    save(prefix, field)





