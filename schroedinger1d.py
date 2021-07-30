import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# $$
# i \hbar \partial_t | \psi \rangle = H | \psi \rangle
# $$
# $$
# \hbar = 1
# $$
# $$
# H = - \frac{1}{2m} \frac{\partial^2}{\partial x^2} + V(x)
# $$
# $$
# V(x) = \frac{V_0}{2}(\cos(2 \pi x / L) + 1), \, \, x \in [-L, L]
# $$
# $$
# x' = x / L
# $$
# $$
# H = - \frac{1}{2m} \frac{\partial^2}{\partial x'^2 L^2} + V(x' L)
#   = - \frac{1}{2m} \frac{\partial^2}{\partial x'^2 L^2} + \frac{V_0}{2}(\cos(2 \pi x') + 1)
# $$
# $$
# H' = 2 m L^2 H = - \frac{\partial^2}{\partial x'^2} + m L^2 V_0 (\cos(2 \pi x') + 1)
# $$
# $$
# \tilde{V} = m L^2 V_0
# $$
# $$
# H' = - \frac{\partial^2}{\partial x'^2} + \tilde{V} \cos(2 \pi x'), \, \, x' \in [-1, 1]
# $$

# In[144]:


N = 100
x_prime = np.linspace(-1, 1, N)
h = x_prime[1] - x_prime[0]
e = np.ones(N)
D2 = sp.diags([1, e[:-1], -2*e, e[:-1], 1], [-N + 1, -1, 0, 1, N - 1]) / h**2 # periodic bounary conditions
V_tilde = 1e2
V = V_tilde * np.cos(2*np.pi*x_prime) # potential
H = - D2 + np.diag(V)
eigen_vals, eigen_vecs = eigsh(H, 4, which="SA")
for i, psi in enumerate(eigen_vecs.T):
    plt.plot(x_prime, psi, label=i)
plt.xlabel(r"$x'$",fontsize=15)
plt.ylabel(r"$\psi$",fontsize=15)
plt.legend()
plt.title(r"$H = - \frac{\partial^2}{\partial x'^2} + \tilde{V} \cos(2 \pi x')$, $\tilde{V} =$" + f"{V_tilde:.2e}")
plt.show()

