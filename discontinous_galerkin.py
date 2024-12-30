# https://www.youtube.com/playlist?list=PLc4eVvgECLtptmgobxxXmVikejqPf4f8D

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import inv
from numba import njit

# strong formulation of a scalar conservation law:
# ==> dq/dt + div_x f(q) =

# weak formulation: multiply with test function phi and integrate over domain:
# ==> integral_element dq/dt phi dx + integral_element div_x f(q) * phi dx
# use product rule and divergence theorem and pullout time derivative (test function time-independent):
# ==> d/dt integral_element q phi dx + integral_element (div_x [f(q) * phi] - f(q) * grad_x phi] dx
# ==> d/dt integral_element q phi dx + integral_surface f(q) * phi dA - integral_element f(q) * grad_x phi dx

# flux function on the boundary becomes numerical flux
# (because q on the boundary is not unique on boundary between elements):
# f -> f^
# we express our solution as a linear combination of basis functions:
# ===> q = sum_i a_i * phi_i
# use the same space for the test functions:
# ===> phi = sum_i a_i * phi_i
# we use <f, g> = integral f(x) g(x) dx:
# in 1D: integral_surface f(q) * phi_j dA = f^_R * phi_j(x_R) - f^_L * phi_j(x_L)

# then forall j:
# ===> sum_i  <phi_j, phi_i> da_i/dt - sum_i c * <dphi_j/dx, phi_i> * a_i  + phi_j(x_R) * f^_R  - phi_j(x_L) * f^_L
#      sum_i dot a_i M_ij - sum_i c * a_i K_ij + f^_left - f^_right = 0

# M_ji = <phi_j, phi_i>
# K_ij = <dphi_j/dx, phi_j>
# f^_R * phi_j(x_R) - f^_L * phi_j(x_L)
# for phi_1(x_R) = 0.0 and phi_1(x_L) = 1.0
# for phi_2(x_L) = 0.0 and phi_2(x_R) = 1.0
# ===> f^_1 = f^_R * phi_1(x_R) - f^_L * phi_1(x_L) = - f^L
#      f^_2 = f^_R * phi_2(x_R) - f^_L * phi_2(x_L) = f^R

# da/dt M - c a K + f = 0
# da/dt = (c a K - f) * M^-1


################################### flux: advection ##############################
c = 1.0
@njit
def flux_function(q, c):
    return c * q

# upwind flux
@njit
def numerical_flux_function(c, q_left, q_right):
    return flux_function(q_left, c) if c > 0.0 else flux_function(q_right, c)

#################################### basis ########################################
# in each element: q = sum_i a_i f_i
x = sp.symbols("x", real=True)
basis = [1 - x, x]

# TODO: generate basis from nodes (any polygom degree for the appropriate number of nodes)

# mass matrix M_ji = integral_element phi_i phi_j
M = np.array([[float(sp.integrate(phi1 * phi2, (x, 0, 1))) for phi1 in basis] for phi2 in basis])
M_inv = inv(M)
# stiffness matrix K_ji = integral_element phi_j / dx phi_i d dx
K = np.array([[float(sp.integrate(phi1 * phi2.diff(x), (x, 0, 1))) for phi1 in basis] for phi2 in basis])

# rhs (semi-discretisation)
n_coeffs_per_element = len(basis)

LEFT = 0
RIGHT = 1

################################## mesh #####################################
# discretisation into finite elements with basis functions on each element
n_elements = 20
elements = np.linspace(0.0, 1.0, n_elements + 1) # position of the boundaries of the elements
dxs = np.diff(elements)
# position of the nodes of each element
element_nodes = np.concatenate([[elements[0]], elements[1:-1].repeat(2), [elements[-1]]])

@njit
def get_coeff(a, element_index, side_index):
    element_index %= n_elements # periodic boundary conditions
    assert 0 <= side_index < n_coeffs_per_element # safety check
    coeff_index = n_coeffs_per_element * element_index + side_index
    return a[coeff_index]

######################## discontious garlekin method ###########################
@njit
def compute_flux(c, a, element_index):
    q_left = get_coeff(a,  element_index - 1, RIGHT)
    q_right = get_coeff(a, element_index,     LEFT)
    f_left = numerical_flux_function(c, q_left, q_right)

    q_left = get_coeff(a,  element_index,     RIGHT)
    q_right = get_coeff(a, element_index + 1, LEFT)
    f_right = numerical_flux_function(c, q_left, q_right)

    return np.array([- f_left, f_right])

@njit
def rhs(t, a):
    dadt = np.zeros_like(a)
    for i in range(n_elements):
        element_a = a[n_coeffs_per_element*i : n_coeffs_per_element*(i + 1)]
        flux = compute_flux(c, a, i)
        # dx comes from the change of coordinates from [0,1] element to [0, dx] element
        # only one dx because we have one from M (divide bc of inverse M)
        # in K dx from the integral cancels the dx from the derivative
        dadt[n_coeffs_per_element*i : n_coeffs_per_element*(i + 1)] = M_inv @ (c * K @ element_a - flux) / dxs[i]
    return dadt

##################################### time stepping ###################################
def initial_condition(x): return np.exp(-(x - 0.5)**2 * 10)
# initial value for the coefficients of a all basis functions
initital_element_coeffs = initial_condition(element_nodes)
safety = 10.0
tmax = 0.1
# solve semi-discirete form using scipy ode solver
sol = solve_ivp(rhs, (0, tmax), initital_element_coeffs, max_step=np.min(dxs) / c / safety) # ensure CFL condition
assert sol.success, "solver failed"
C_CFL = 1.0
assert np.all(np.min(c / dxs) > C_CFL * np.max(np.diff(sol.t))), "CFL condition violated by solver"
final_element_coeffs = sol.y[:, -1]

# plot
plt.figure()
plt.plot(elements, initial_condition(elements), label="initial")
plt.plot(elements, initial_condition((elements - c * tmax) % (elements[-1] - elements[0])), label="analytical solution")
plt.plot(element_nodes, final_element_coeffs, "x", label="dg solution")
plt.xlabel("x")
plt.ylabel("q(x)")
plt.legend()
plt.show()
