# based on: https://www.youtube.com/watch?v=Kn5P_1-cjoE
import numpy as np, matplotlib.pyplot as plt
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.special import erf

# spacial discretisation
g = 20
p = np.linspace(-3, 3, g)
x, y, z = np.meshgrid(p, p, p)
h = p[1] - p[0]
x, y, z = x.ravel(), y.ravel(), z.ravel()

# external potential from nucleus
R = np.sqrt(x**2 + y**2 + z**2)
Vext = -2/R

# kinetic energy operator in 3D
e = np.ones(g)
L = sp.diags([e[1:], -2*e, e[:-1]], [-1,0,1]) / h**2
I = sp.eye(g)
L3 = sp.kron(sp.kron(L, I), I) + sp.kron(sp.kron(I, L), I) + sp.kron(sp.kron(I, I), L)

# compensation charges
n_comp = np.exp(-R**2 / 2)
n_comp = -2 * n_comp / (np.sum(n_comp)*h**3)
V_comp = -2 / R * erf(R / np.sqrt(2))

# co-Hamiltonian guess
Vtot = Vext

# self consistent loop
while True:
    # solve Kohn Sham equations
    H = -0.5*L3 + sp.diags(Vtot, 0)
    E, PSI = spla.eigsh(H, 1, which="SA")

    # electron density
    PSI = PSI[:, 0] / h**(3/2) # normalize to electrons per unit volume
    n = 2*np.abs(PSI)**2

    # ????
    Vx = -(3/4)*(3/np.pi*n)**(1/3)
    # Hartree potential, field from electron density
    Vh, status = spla.cgs(L3, -4*np.pi*(n + n_comp), tol=1e-7, maxiter=400)
    assert status == 0
    Vh -= V_comp
    Vtot = Vx + Vh + Vext

    # kinetic energy and expectation values of the energies
    T = 2 * PSI.T.conj() @ (-0.5 * L3) @ PSI * h**3
    Eext = np.sum(n * Vext)*h**3
    Eh = 0.5 * np.sum(n * Vh)*h**3
    Ex = np.sum(n * Vx)*h**3
    Etot = T + Eext + Eh + Ex

    print("E =", Etot)
