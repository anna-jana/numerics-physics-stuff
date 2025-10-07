import os.path
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.optimize import curve_fit
from numba import njit

############################ SU(3) group #########################
@njit
def adjoint(A):
    return A.T.conj()

@njit
def reproject_su3(A):
    A[0, :] /= np.linalg.norm(A[0, :])
    A[1, :] -= A[0, :] * np.dot(A[0, :], A[1, :])
    A[1, :] /= np.linalg.norm(A[1, :])
    A[2, :] = np.cross(A[0, :], A[1, :])

def random_su2():
    I = np.eye(2, dtype=np.complex128)

    pauli = np.array([[[0, 1],
                       [1, 0]],
                      [[0, -1j],
                       [1j, 0]],
                      [[1, 0],
                       [0, -1]]], dtype=np.complex128)

    r = np.random.rand(4) - 1.0
    eps = 1e-3
    x = eps * r[1:] / np.linalg.norm(r[1:])
    x0 = np.sign(r[0]) * np.sqrt(1 - eps**2)

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
@njit
def get_link(field, i, j, k, l, d, direction):
    N = field.shape[0]
    Nt = field.shape[3]
    # link in -1 direction on axis d from node {A_n}
    #                   is the same as
    # link in +1 direction on axis d from neighboring node in -1 dieection on axis d
    # (excluding the adjoint)
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

@njit
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

######################################### physics ######################################
@njit
def compute_action_diff(beta, field, i, j, k, l, d, new):
    N = field.shape[0]
    A = np.zeros((3, 3), dtype=np.complex128)
    for d_offset in range(4):
        if d_offset != d:
            for offset_direction in (-1, +1):
                #                    ^              U2
                #                    |          +-----+
                # d_offset, diection |   U1 --> |     | <-- U3
                #                    | (i,j,k,l)+=====+
                #                    |             ^
                #                    |         changed link
                #                    +--------------------->
                #                        d, always in +1 direction
                U1 = adjoint(get_link(field, i, j, k, l, d_offset, offset_direction))
                U2 = adjoint(get_link_with_offset(field, i, j, k, l, d_offset, offset_direction, d, 1))
                U3 = get_link_with_offset(field, i, j, k, l, d, 1, d_offset, offset_direction)
                A += U1 @ U2 @ U3
    N = field.shape[0]
    return - beta / N * np.real(np.trace((new - field[i, j, k, l, d, +1]) @ A))

@njit
def plaquetts(field):
    N = field.shape[0]
    Nt = field.shape[3]
    plaq = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(Nt):
                    for d1 in range(4):
                        for d2 in range(d1):
                            #        ^             U2
                            #        |          +-->--+
                            #        |          |     |
                            # d2, +1 |       U1 ^     v U3
                            #        |          |     |
                            #        | (i,j,k,l)+--<--+
                            #        |             U4
                            #        +--------------------->
                            #                d1, +1
                            U1 = get_link(field, i, j, k, l, d2, +1)
                            U2 = get_link_with_offset(field, i, j, k, l, d2, +1, d1, +1)
                            U3 = adjoint(get_link_with_offset(field, i, j, k, l, d1, +1, d2, +1))
                            U4 = adjoint(get_link(field, i, j, k, l, d1, +1))
                            plaq += np.real(np.trace(U1 @ U2 @ U3 @ U4))
                            count += 1
    return plaq / count

################################### algorithm ##############################
@njit
def reproject_all(field):
    N = field.shape[0]
    Nt = field.shape[3]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(Nt):
                    for d in range(4):
                        reproject_su3(field[i, j, k, l, d, :, :])

def run_simulation(beta, field, nsteps):
    N = field.shape[0]
    Nt = field.shape[3]
    reproject_every = 10
    ps = []
    for mc_step in tqdm.tqdm(range(nsteps)):
        # mc sweep
        for _ in range(N**3 * Nt):
            # metropolis
            i = np.random.randint(N)
            j = np.random.randint(N)
            k = np.random.randint(N)
            l = np.random.randint(Nt)
            d = np.random.randint(4)
            new = random_su3()
            action_diff = compute_action_diff(beta, field, i, j, k, l, d, new)
            if np.exp(-action_diff) <= np.random.rand():
                field[i, j, k, l, d, :, :] = new
        if mc_step % reproject_every == 0:
            reproject_all(field)
        # observables
        ps.append(plaquetts(field))
    return ps

########################################## data analysis ##########################
def auto_correlation(ps):
    C = []
    N = len(ps)
    for n in range(N - 1):
        L = ps[n:]
        R = ps[:N - n]
        C.append(np.mean(L * R) - np.mean(L) * np.mean(R))
    return np.array(C)

def plot(prefix):
    ps = np.loadtxt(os.path.join(prefix, "plaquetts.dat"))

    C = auto_correlation(ps)
    f = lambda steps, scale, tau: scale * np.exp(- steps / tau)
    (scale, tau), _ = curve_fit(f, np.arange(C.size), C, p0=(1.0, 1.0))

    plt.figure()
    plt.plot(C, label="simulation data")
    plt.plot(f(np.arange(C.size), scale, tau), label="expoential fit")
    plt.xlabel("mc step")
    plt.ylabel("auto correlation of mean plaquett value")
    plt.legend()
    plt.title(f"{tau = }")
    plt.show()

################################# io and user interface ############################
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
    np.save(os.path.join(prefix, "field"), field)
    with open(os.path.join(prefix, "plaquetts.dat"), "w") as fh:
        fh.write("")

def run(prefix, nsteps):
    name = os.path.join(prefix, "field.npy")
    field = np.load(name)
    with open(os.path.join(prefix, "beta.txt"), "r") as fh:
        beta = float(fh.read())
    ps = run_simulation(beta, field, nsteps)
    np.save(os.path.join(prefix, "field"), field)
    with open(os.path.join(prefix, "plaquetts.dat"), "a") as fh:
        fh.write("\n" + "\n".join(map(str, ps)))

