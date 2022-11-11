import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1996)

class MolecularDynamicsSimulation:
    def __init__(self, x, v, L, T, m):
        self.x = x
        self.v = v
        self.t = 0
        self.L = L
        self.T = T
        self.m = m
        self.a = self.calc_a(self.x)

    @staticmethod
    def maxwell_velocity(T, m, shape):
        sigma_v = np.sqrt(T / m)
        return np.random.normal(0, sigma_v, shape)

    @classmethod
    def make_particle_box(cls, N, V, T, m=1.0, d=3):
        L = np.cbrt(V)
        x = np.random.uniform(0, L, (N, d))
        v = cls.maxwell_velocity(T, m, (N, d))
        return cls(x, v, L, T, m)

    def velocity_verlet_step(self, dt):
        new_x = self.x + dt*self.v + dt**2*self.a
        new_a = self.calc_a(new_x)
        self.v += 0.5*dt*(self.a + new_a)
        self.x = new_x
        self.a = new_a

    def reflect_on_boundaries(self):
        is_outside = self.x < 0.0
        self.v[is_outside] *= -1.0
        self.x[is_outside] *= -1.0

        is_outside = self.x > self.L
        self.v[is_outside] *= -1.0
        self.x[is_outside] *= self.L - (self.x[is_outside] - self.L)

    def andereson_thermostat_step(self, vv_dt, coupling_strength):
        dt = np.random.exponential(1 / coupling_strength)
        for i in range(int(dt / vv_dt)):
            self.velocity_verlet_step(vv_dt)
        i = np.random.randint(self.x.shape[0])
        self.v[i, :] = self.maxwell_velocity(self.T, self.m, (self.x.shape[1],))

    def mc_step(self, vv_dt, coupling_strength):
        for i in range(self.x.shape[0]):
            self.andereson_thermostat_step(vv_dt, coupling_strength)

    def run_simulation(self, num_mc_steps, vv_dt, coupling_strength):
        timeseries = np.empty(num_mc_steps)
        for i in range(num_mc_steps):
            self.mc_step(vv_dt, coupling_strength)
            timeseries[i] = self.meaasure()
        return timeseries

class IdealGasSimulation(MolecularDynamicsSimulation):
    def calc_a(self, x):
        return np.zeros_like(x)

    def meaasure(self):
        # kinetic theory temperature
        K = sum(0.5*self.m*np.linalg.norm(v)**2 for v in self.v)
        N = self.x.shape[0]
        return 2/3 * K/N

s = IdealGasSimulation.make_particle_box(N=100, V=1.0, T=2.0)
Ts = s.run_simulation(400, 0.1, 10.0)
plt.plot(Ts)
plt.xlabel("mc step")
plt.ylabel("T / code units")
plt.title("MD sim with Anderson thermostat for ideal gas")
plt.show()
