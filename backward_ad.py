from dataclasses import dataclass
from typing import Union, List, Callable
from numbers import Real
import math
from functools import reduce

@dataclass
class Var:
    name: str
    val: Real
    diff: Union[List[Real], None] = None

    def reset(self):
        self.val = self.diff = None

    def get_val(self):
        return self.val

    def get_diff(self, parameters):
        if self.diff is None:
            self.diff = [int(p is self) for p in parameters]
        return self.diff

    def __str__(self):
        return self.name

@dataclass
class Const:
    val: Real

    def reset(self):
        self.val = None

    def get_val(self):
        return self.val

    def get_diff(self, parameters):
        return [0.0]*len(parameters)

    def __str__(self):
        return str(self.val)

@dataclass
class Op:
    name: str
    args: List[Union[Var, Const, "Op"]]
    f: Callable
    df: Callable
    val: Union[Real, None] = None
    diff: Union[List[Real], None] = None

    def reset(self):
        self.val = self.diff = None
        for a in self.args:
            a.reset()

    def get_val(self):
        if self.val is None:
            self.val = self.f(*[a.get_val() for a in self.args])
        return self.val

    def get_diff(self, parameters):
        if self.diff is None:
            # d f(g_1(x_1, ..., x_n), g_2(x_1, ..., x_n))/dx_i = df/dx_j * dg_j/dx_i
            inner = [a.get_diff(parameters) for a in self.args]
            outer = self.df(*[a.get_val() for a in self.args])
            self.diff = [sum(outer[j] * inner[j][i] for j in range(len(self.args))) for i in range(len(parameters))]
        return self.diff

    def __str__(self):
        return self.name + "(" + ", ".join([str(a) for a in self.args]) + ")"


def add(x, y):
    return Op("+", [x, y], lambda a, b: a + b, lambda a, b: [1.0, 1.0])

def mul(x, y):
    return Op("*", [x, y], lambda a, b: a*b, lambda a, b: [b, a])

def power(x, y):
    # x^y = exp(log(x)*y)
    # d/dx x^y = y*exp(log(x)*y)/x = y*x^y / x = yx^(y - 1)
    # d/dy x^y = log(x)*exp(log(x)*y) = log(x)*x^y
    return Op("pow", [x, y], lambda a, b: a**b, lambda a, b: [b*a**(b - 1), math.log(a)*a**b])

def divide(x, y):
    return Op("/", [x, y], lambda a, b: a / b, lambda a, b: [1/b, -a / b**2])

x = Var("x", 1.0)
y = reduce(add, [divide(power(x, Const(n)), Const(math.factorial(n))) for n in range(10)])
print("expression:", y)
print("value:", y.get_val())
print("derivative:", y.get_diff([x]))
