from pylab import *
from scipy.integrate import solve_ivp

def H_to_t(H): return 1 / (2*H)
def t_to_H(t): return 1 / (2*t)

def rhs(t, u,  m1, m2, g):
    H = H_to_t(t)
    phi1, phi2, phi1_dot, phi2_dot = u
    phi1_dot_dot = - 3 * H * phi1_dot - m1**2 * phi1 - g * phi1 * phi2**2
    phi2_dot_dot =  - 3 * H * phi2_dot - m2**2 * phi2 - g * phi2 * phi1**2
    return (phi1_dot, phi2_dot, phi1_dot_dot, phi2_dot_dot)

H0 = 100.0
H1 = 1e-5
f1 = 1.0
f2 = 2.0
m1 = 0.02
m2 = 0.01
g = 10.0

t0 = H_to_t(H0)
t1 = H_to_t(H1)

sol = solve_ivp(rhs, (t0, t1), (f1, f2, 0.0, 0.0), args=(m1, m2, g))

figure()
plot(sol.t, sol.y[0, :] / f1, label=r"$\phi_1$")
plot(sol.t, sol.y[1, :] / f2, label=r"$\phi_2$")
xscale("log")
legend()
xlabel(r"$t$ / a.u.")
ylabel("field / initial value")
title(r"$V = \frac{m_1^2}{2} \phi_1^2 + \frac{m_2^2}{2} \phi_2^2 + \frac{g}{2} \phi_1^2 \phi_2^2$" +
      f"\n\$m_1 = \$ {m1}, \$m_2 = \$ {m2}, g = {g}, \$\\phi_1(t_0)\$ = {f1}, \$\\phi_2(t_0)\$ = {f2}")
tight_layout()
show()
