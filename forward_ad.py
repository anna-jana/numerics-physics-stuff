import numpy as np
import matplotlib.pyplot as plt

class Dual:
    def __init__(self, a, b=0):
        if isinstance(a, Dual):
            assert b == 0
            self.a = a.a
            self.b = a.b
        else:
            self.a = float(a)
            self.b = float(b)

    # (a1 + b1 eps) + (a2 + b2 eps) = (a1 + a2) + (b1 + b2) eps
    def __add__(self, other):
        other = Dual(other)
        return Dual(self.a + other.a, self.b + other.b)

    def __radd__(self, other): return self + other

    def __sub__(self, other):
        other = Dual(other)
        return Dual(self.a - other.a, self.b - other.b)

    def __rsub__(self, other): return self - other

    def __neg__(self):
        return Dual(-self.a, -self.b)

    # (a1 + b1 eps) * (a2 + b2 eps) = a1 * a2 + (b1 * a2 + b2 * a1) eps
    def __mul__(self, other):
        other = Dual(other)
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __rmul__(self, other): return self * other

    # (a1 + b1 eps) / (a2 + b2 eps) = (a1 + b1 eps) (a2 - b2 eps) / [(a2 + b2 eps) (a2 - b2 eps)]
    # = [a1 a2 + (a2 b1 - a1 b2) eps] / [a2 a2 + b2 a2 eps - b2 a2 eps]  =
    def __truediv__(self, other):
        other = Dual(other)
        return Dual(self.a / other.a, (other.a * self.b - other.b * self.a) / other.a**2)

    def __rdiv__(self, other): return self / other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        return Dual(self.a**other.a,  other * self.a**(other - 1) * self.b)

    def __le__(self, other): return self.a <= Dual(other).a
    def __lt__(self, other): return self.a <  Dual(other).a
    def __ge__(self, other): return self.a >= Dual(other).a
    def __gt__(self, other): return self.a >  Dual(other).a
    def __eq__(self, other): return self.a == Dual(other).a
    def __ne__(self, other): return self.a != Dual(other).a

    def sign(self):
        return Dual(np.sign(self.a))

    def __abs__(self):
        return Dual(np.abs(self.a), self.b * np.sign(self.a))

    def __mod__(self, other):
        assert isinstance(other, (int, float))
        return Dual(np.fmod(self.a, other), self.b)

def simple_sin(x, tol = 1e-10):
    # map to 0..inf range
    sign = x.sign()
    x = abs(x)
    # map to 0..2pi range
    x = x % (2*np.pi)
    # map to 0..pi range
    if x >= np.pi:
        sign = -sign
        x -= np.pi
    # approximate by power series
    ans = 0.0
    s = 1
    n = 1
    fac = 1
    x_pow = x
    x_sq = x*x
    while True:
        term = s * x_pow / fac
        if abs(term) < tol:
            break
        x_pow *= x_sq
        fac *= (n + 2) * (n + 1)
        n += 2
        s = -s
        ans += term
    # multiply sign back in
    return sign * ans

if __name__ == "__main__":
    xs = np.linspace(-2*np.pi, 2*np.pi, 500)
    ys = [simple_sin(Dual(x, 1.0)) for x in xs]
    evals = [y.a for y in ys]
    derives = [y.b for y in ys]
    plt.plot(xs, evals, label="eval sin(x)")
    plt.plot(xs, derives, label="derivative sin(x) = cos(x)")
    plt.ylabel("x")
    plt.legend()
    plt.show()

