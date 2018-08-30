
# coding: utf-8

# # Solving nonlinear equations (finding roots of functions)
# Find $x$ such that $f(x) = 0$

import numpy as np
import matplotlib.pyplot as plt

tol = 1e-10
max_steps = 100
f = lambda x: x**2 - 2.0
df = lambda x: 2*x
print("root of x^2 - 2:", np.sqrt(2))

# ## Newtons method
# From:
# $$ f(x^*) = 0 = f(x_0) + f'(x_0)(x^* - x_0) + \mathcal{O}(|x^* - x_0|^2) $$
# we get:
# $$ x_{n+1} \leftarrow x_n - \frac{f(x_n)}{f'(x_n)}$$

def newton(f, df, x0, tol=tol, max_steps=max_steps):
    x = x0
    step = 0
    while True:
        x_new = x - f(x)/df(x)
        if abs(x - x_new) <= tol or step == max_steps:
            return x_new, step
        x = x_new
        step += 1

print("newton:", newton(f, df, 10.0))

# ## Bisection
# We have two points $a$ and $b$. Now we compute the point between thous $c = \frac{a + b}{2}$. If $f(c)$ has the same sign as $f(a)$ then we can use $c$ as our new $a$, if $f(c)$ as the same sign as $f(b)$ we can use $c$ as our new $b$.

def shrink_interval(f, a, b, c):
    if np.sign(f(c)) == np.sign(f(a)):
        return c, b
    else:
        return a, c

def bisect(f, a, b, tol=tol, max_steps=max_steps):
    step = 0
    assert np.sign(f(a)) != np.sign(f(b))
    while True:
        c = (a + b)/2
        if abs(a - b) <= tol or step == max_steps or f(c) == 0.0:
            return c, step
        a, b = shrink_interval(f, a, b, c)
        step += 1


print("bisection:", bisect(f, 0.0, 10.0, tol=1e-15, max_steps=200))


# ## Secant method
# We have two points $a$ and $b$. Then we can approx. the root between these points by the root of the secant between the points.
# The secant hits $(a, f(a))$ and $(b, f(b))$.
# $$m = \frac{f(a) - f(b)}{a - b}$$
# $$b = f(a) - m a$$
# $$ 0 = mx + b \Rightarrow x = -\frac{b}{m}$$

def secant_root(f, a, b):
    m = (f(a) - f(b))/(a - b)
    d = f(a) - m*a
    return -d/m

def secant(f, x0, x1, tol=tol, max_steps=max_steps):
    step = 0
    while True:
        x2 = secant_root(f, x0, x1)
        if abs(x1 - x2) <= tol or step == max_steps:
            return x2, step
        x0, x1 = x1, x2
        step += 1

print("secant:", secant(f, 0.0, 10.0))

# ## Regula falsi (false position)
# Lets combine secant and bisection!
#
# Start with an interval $(a_0,b_0)$ where $sgn(f(a_0)) \neq sgn(f(b_0))$
# - Find root of the secant $c$ between $a_0$ and $b_0$
# - We are done if $f(c) = 0$ or we reached some tolerance.
# - Choose $a_1 = c$ if $sgn(f(c)) = sgn(f(a))$ otherwise choose $b_1 = c$ if $sgn(f(c)) = sgn(f(b))$

def regula_falsi(f, a, b, tol=tol, max_steps=max_steps):
    assert np.sign(a) != np.sign(b)
    step = 0
    while True:
        c = secant_root(f, a, b)
        if min(abs(a - c), abs(b - c)) <= tol or c == 0.0 or step == max_steps:
            return c, step
        a, b = shrink_interval(f, a, b, c)
        step += 1

print("regula_falsi:", regula_falsi(f, 0.0, 10.0))

# ## Fixpoint iteration
# Instead of solving $f(x) = 0$ for $x$, we turn this equation into $g(x) = x$ and then compute $x_{n + 1} = g(x_n)$ for some sufficently large $n$.

def fixpoint(g, x, tol=tol, max_steps=max_steps, debug=False):
    step = 0
    while True:
        new_x = g(x)
        if debug:
            print(x, new_x)
        if abs(x - new_x) <= tol or step == max_steps:
            return new_x, step
        x = new_x
        step += 1

# $$f(x) = x^2 - 2$$
# $$x^2 - 2 = 0$$
# $$g(x) = x^2 + x - 2$$
# $$x = x^2 + x - 2$$

fixpoint(lambda x: x**2 + x - 2., 1.4, max_steps=10, debug=True)

# $$f(x) = sin(x)$$
# $$sin(x) = 0$$
# $$g(x) = sin(x) + x$$
# $$x = sin(x) + x$$

# In[15]:

fixpoint(lambda x: np.sin(x) + x, 1.0, debug=True)
