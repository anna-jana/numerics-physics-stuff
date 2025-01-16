import numpy as np
import matplotlib.pyplot as plt

Z = np.arange(1, 200)
N = np.arange(1, 200)
N, Z = np.meshgrid(N, Z)
A = Z + N

# all in MeV
a_V = 15.67
a_surf = 17.23
a_c = 0.714
a_s = 93.15
a_p = 11.2

E_vol = a_V * A
E_surf = a_surf * A**(2/3)
E_coloumb = a_c * Z * (Z - 1) * A**(-1/3)
E_sym = a_s * (N - Z)**2 / (4*A)
E_pair = (np.where((Z % 2 == 0) & (N % 2 == 0), +1,
   np.where((Z % 2 == 1) & (N % 2 == 1), -1, 0))) * a_p * A**(-1/2)
E = E_vol - E_surf - E_coloumb - E_sym + E_pair

plt.figure()
plt.pcolormesh(N, Z, np.where(E > 0, E / A, np.NaN))
plt.colorbar(label="Binding energy per nucleon E / A / MeV")
plt.xlabel("Neutron number, N")
plt.ylabel("Proton number, Z")
plt.title("Bethe Weizaecker formula for nuclear binding energy")
plt.show()
