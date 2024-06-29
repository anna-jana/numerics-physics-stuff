import sympy as sp
import numpy as np

class SphericalHarmomic:
    def __init__(self, l: int, m: int, N: float):
        assert l >= 0 and abs(m) <= l
        self.l = l
        self.m = m
        self.N = N
        theta = sp.symbols("theta")
        self.P = sp.assoc_legendre(l, m, sp.cos(theta))
        self.P_fn = sp.lambdify((theta,), self.P)

    def __call__(self, theta, phi):
        return self.N * np.exp(1j * self.m * phi) * self.P_fn(theta)

