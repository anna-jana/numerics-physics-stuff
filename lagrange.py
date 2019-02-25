import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# parameter
k, m = sp.symbols("k m")

# time
t = sp.Symbol("t")

# generalized coordinate
q_fn = sp.Function("q")
q = q_fn(t)
q_dot = q.diff(t)

# cartesian coordinates in terms of q
x = q
y = 0
z = 0
print("generalized coordinates:")
print("x =", x, "\ny =", y, "\nz =", z)

# compute cartesian derivates in terms of q
x_dot = sp.diff(x, t)
y_dot = sp.diff(y, t)
z_dot = sp.diff(z, t)
print("x_dot =", x_dot, "\ny_dot =", y_dot, "\nz_dot =", z_dot)

# construct lagrangian
T = m*(x_dot**2 + y_dot**2 + z_dot**2)/2
V = k*x**2/2
L = T - V
print("lagrangian:", L)

# compute lagrange equation
lagrange_eq = sp.Eq(L.diff(q_dot).diff(t) - L.diff(q), 0)
print("equation of motion:", lagrange_eq)

# general solution
q_general_sol = sp.dsolve(lagrange_eq, q).rhs
q_dot_general_sol = q_general_sol.diff(t)
print("solution:")
print("q(t) =", q_general_sol)
print("q_dot(t) =", q_dot_general_sol)
C1, C2 = sp.symbols("C1 C2")

# solve initial conditions
q0 = 0
q_dot_0 = 1
eq1 = sp.Eq(q_general_sol.subs(t, 0), q0)
eq2 = sp.Eq(q_dot_general_sol.subs(t, 0), q_dot_0)
sol = sp.solve([eq1, eq2], [C1, C2])
print("parameters for the initial conditions:")
print("C1 =", sol[C1])
print("C2 =", sol[C2])
q_sol = q_general_sol.subs(C1, sol[C1]).subs(C2, sol[C2])
q_dot_sol = q_sol.diff(t)
print("solutions for initial conditions:")
print("q(t) =", q_sol)
print("q_dot(t) =", q_dot_sol)

# choose parameters
time = np.linspace(0, 10, 500)
k_val = 1.0
m_val = 1.0
q_vals = sp.lambdify(t, q_sol.subs(k, k_val).subs(m, m_val))(time).real
plt.plot(time, q_vals)
plt.xlabel("t [s]")
plt.ylabel("q [m]")
plt.title("q(0) = %.2f, q_dot(0) = %.2f, k = %.2f, m = %.2f" % (q0, q_dot_0, k_val, m_val))
plt.grid()
plt.show()
