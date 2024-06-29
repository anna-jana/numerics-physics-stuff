# Hartree-Fock

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.special import erf
from scipy.linalg import eigh, norm

@dataclass
class Nucleus:
    center: NDArray[float]
    charge: int

def F0(x):
    if x == 0.0:
        return 1.0
    return x**(-0.5) * np.sqrt(np.pi) / 2 * erf(x**0.5)

@dataclass
class GTO_1S:
    center: NDArray[float]
    zeta: float

    @staticmethod
    def overlap(phi1: "GTO_1S", phi2: "GTO_1S") -> float:
        d2 = norm(phi1.center - phi2.center)**2
        A = phi1.zeta * phi2.zeta / (phi1.zeta + phi2.zeta)
        return (np.pi / (phi1.zeta + phi2.zeta))**(3/2) * np.exp(- A * d2)

    @staticmethod
    def kinetic(phi1: "GTO_1S", phi2: "GTO_1S") -> float:
        d2 = norm(phi1.center - phi2.center)**2
        A = phi1.zeta * phi2.zeta / (phi1.zeta + phi2.zeta)
        S = (np.pi / (phi1.zeta + phi2.zeta))**(3/2) * np.exp(- A * d2)
        return A * (6 - 4 * A * d2) * S

    @staticmethod
    def nuclear_attraction(nuclei: list[Nucleus]):
        def _nuclear_attraction(phi1: "GTO_1S", phi2: "GTO_1S"):
            d2 = norm(phi1.center - phi2.center)**2
            A = phi1.zeta * phi2.zeta / (phi1.zeta + phi2.zeta)
            R_P = (phi1.zeta * phi1.center + phi2.zeta * phi2.center) / (phi1.zeta + phi2.zeta)
            return sum(- 2*np.pi * nucleus.charge / (phi1.zeta + phi2.zeta) *
                       np.exp(- A * d2) *
                       F0((phi1.zeta + phi2.zeta) * norm(R_P - nucleus.center)**2)
                for nucleus in nuclei)
        return _nuclear_attraction

    @staticmethod
    def two_electron_integral(phi_A: "GTO_1S", phi_B: "GTO_1S", phi_C: "GTO_1S", phi_D: "GTO_1S"):
        helper1 = (
            (phi_A.zeta + phi_C.zeta) *
            (phi_B.zeta + phi_D.zeta) *
            (phi_A.zeta + phi_B.zeta + phi_C.zeta + phi_D.zeta)**0.5
        )
        helper2 = (
            (phi_A.zeta + phi_C.zeta) * (phi_B.zeta + phi_D.zeta) /
            (phi_A.zeta + phi_B.zeta + phi_C.zeta + phi_D.zeta)
        )
        d2_AC = norm(phi_A.center - phi_C.center)**2
        d2_BD = norm(phi_B.center - phi_D.center)**2
        R_P = (phi_A.zeta * phi_A.center + phi_C.zeta * phi_C.center) / (phi_A.zeta + phi_C.zeta)
        R_Q = (phi_B.zeta * phi_B.center + phi_D.zeta * phi_D.center) / (phi_B.zeta + phi_D.zeta)
        d2_PQ = norm(R_P - R_Q)**2
        return (
                2*np.pi**(5/2) / helper1 *
                np.exp(- phi_A.zeta * phi_C.zeta / (phi_A.zeta + phi_C.zeta) * d2_AC
                       - phi_B.zeta * phi_D.zeta / (phi_B.zeta + phi_D.zeta) * d2_BD) *
                F0(helper2 * d2_PQ)
        )

@dataclass
class CGTO_1S:
    coefficients: NDArray[float]
    gaussians: list[GTO_1S]

    @staticmethod
    def compute_cgto_1s_one_electron_matrix_element(compute_single_gaussian_matrix_element, cgto_1s_1: "CGTO_1S", cgto_1s_2: "CGTO_1S"):
        return sum(c1 * c2 * compute_single_gaussian_matrix_element(g1, g2)
            for c1, g1 in zip(cgto_1s_1.coefficients, cgto_1s_1.gaussians)
                for c2, g2 in zip(cgto_1s_2.coefficients, cgto_1s_2.gaussians))

    @staticmethod
    def overlap(phi1, phi2):
        return CGTO_1S.compute_cgto_1s_one_electron_matrix_element(GTO_1S.overlap, phi1, phi2)

    @staticmethod
    def kinetic(phi1, phi2):
        return CGTO_1S.compute_cgto_1s_one_electron_matrix_element(GTO_1S.kinetic, phi1, phi2)

    @staticmethod
    def nuclear_attraction(nuclei):
        return (lambda phi1, phi2:
            CGTO_1S.compute_cgto_1s_one_electron_matrix_element(GTO_1S.nuclear_attraction(nuclei), phi1, phi2))

    @staticmethod
    def two_electron_integral(cgto_1s_1: "CGTO_1S", cgto_1s_2: "CGTO_1S", cgto_1s_3: "CGTO_1S", cgto_1s_4: "CGTO_1S"):
        return sum(c1 * c2 * c3 * c4 * GTO_1S.two_electron_integral(g1, g2, g3, g4)
            for c1, g1 in zip(cgto_1s_1.coefficients, cgto_1s_1.gaussians)
                for c2, g2 in zip(cgto_1s_2.coefficients, cgto_1s_2.gaussians)
                    for c3, g3 in zip(cgto_1s_3.coefficients, cgto_1s_3.gaussians)
                        for c4, g4 in zip(cgto_1s_4.coefficients, cgto_1s_4.gaussians))

class HartreeFockSolver:
    @staticmethod
    def compute_one_electron_matrix(compute_matrix_element, basis):
        return np.array([[compute_matrix_element(phi1, phi2)
        for phi1 in basis] for phi2 in basis])

    @staticmethod
    def compute_two_electron_matrix(basis):
        K = len(basis)
        BasisFunction = type(basis[0])
        two_electron = -np.ones((K, K, K, K)) # NOTE: setting this to -1 to catch any uninitizalzed elements

        for p in range(K):
            for q in range(p + 1):
                for r in range(p - 1 + 1):
                    for s in range(r + 1):
                        two_electron[p, r, q, s] = two_electron[q, r, p, s] = two_electron[p, s, q, r] = two_electron[q, s, p, r] = \
                                two_electron[r, p, s, q] = two_electron[s, p, r, q] = two_electron[r, q, s, p] = two_electron[s, q, r, p] = \
                                    BasisFunction.two_electron_integral(basis[p], basis[r], basis[q], basis[s])
                r = p
                for s in range(q + 1):
                    two_electron[p, r, q, s] = two_electron[q, r, p, s] = two_electron[p, s, q, r] = two_electron[q, s, p, r] = \
                            two_electron[r, p, s, q] = two_electron[s, p, r, q] = two_electron[r, q, s, p] = two_electron[s, q, r, p] = \
                                BasisFunction.two_electron_integral(basis[p], basis[r], basis[q], basis[s])

        return two_electron - 0.5 * np.swapaxes(two_electron, 2, 3)


    def __init__(self, basis, nuclei):
        BasisFunction = type(basis[0])
        self.basis = basis
        self.nuclei = nuclei

        # one electron integrals
        self.S = HartreeFockSolver.compute_one_electron_matrix(BasisFunction.overlap, basis)
        self.T = HartreeFockSolver.compute_one_electron_matrix(BasisFunction.kinetic, basis)
        self.N = HartreeFockSolver.compute_one_electron_matrix(BasisFunction.nuclear_attraction(nuclei), basis)
        self.h = self.T + self.N

        # two electron integrals
        self.g = self.compute_two_electron_matrix(basis)

    def solve(self, eps=1e-10, alpha=0.5, density_matrix_guess=None):
        # density matrix initial guess
        if density_matrix_guess is not None:
            P = density_matrix_guess
        else:
            P = np.zeros((len(self.basis), len(self.basis)))
        step = 1

        #breakpoint()
        while True:
            breakpoint()
            # compute fock operator
            G = np.sum(P[None, :, None, :] * self.g, axis=(1, 3))
            F = self.h + G
            # solve roothann euqation
            e, C = eigh(F, self.S)
            # normalize
            C /= np.sqrt(np.diag(C.T @ self.S @ C))[None, :]
            # C /= np.sum(C.T @ self.S @ C)
            # new density matrix
            new_P = 2 * C @ C.T
            # check convergence
            delta = norm(P - new_P)
            print(f"{step = }, {delta = }")
            if delta < eps:
                break
            step += 1
            P = new_P # P = alpha * P + (1 - alpha) * new_P

        E = 0.5 * (np.sum(self.h * P) + np.sum(e))
        print(f"{E = }")
        return P, e, E


# https://github.com/nickelandcopper/HartreeFockPythonProgram/blob/main/Hartree_Fock_Program.ipynb

energies = []
d_min = 0.01
d_max = 10.0
nsamples = 50
bounds_lengths = np.linspace(d_min, d_max, nsamples)

for d in bounds_lengths:
    nuclei = [Nucleus(np.array((0,0,0)), 1), Nucleus(np.array((d, 0, 0)), 1)]

    zeta_a = 0.3425250914E+01
    zeta_b = 0.6239137298E+00
    zeta_c = 0.1688554040E+00

    H1_pg1a = GTO_1S(nuclei[0].center, zeta_a)
    H1_pg1b = GTO_1S(nuclei[0].center, zeta_b)
    H1_pg1c = GTO_1S(nuclei[0].center, zeta_c)

    H2_pg1a = GTO_1S(nuclei[1].center, zeta_a)
    H2_pg1b = GTO_1S(nuclei[1].center, zeta_b)
    H2_pg1c = GTO_1S(nuclei[1].center, zeta_c)

    coefficients = np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00])

    H1_1s = CGTO_1S(coefficients, [H1_pg1a, H1_pg1b, H1_pg1c])
    H2_1s = CGTO_1S(coefficients, [H2_pg1a, H2_pg1b, H2_pg1c])

    solver = HartreeFockSolver([H1_1s, H2_1s], nuclei)
    print(f"{d = }")
    _, hartree_levels, energy = solver.solve()

    energies.append(energy)

plt.figure()
plt.plot(bounds_lengths, energies)
plt.xlabel(r"bond length in Bor radii $d [a_0]$")
plt.ylabel(r"bond energy in Hartree $E [E_H]$")
plt.show()
