import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as a

gamma = 1.3 # adiabatic exponent
rho_E = 0.999
N = 120 + 1
tmax = 2.0
dt = 1e-4

dx = 1.0 / (N - 1)
nsteps = int(np.ceil(tmax / dt))

x = np.linspace(0.0, 1.0, N)
prefactor = np.where(x < 0.5, 2.2, 0.2)
A = 1 + prefactor * (x - 0.5)**2
dAdx = prefactor * 2 * (x - 0.5)
d_ln_A_dx = dAdx / A

rho = np.ones(N) * rho_E
v = np.zeros(N)
T = np.ones(N)
rho[0] = 1.0
rho[-1] = rho_E

def diff(is_left, f):
    df = (f[1:] - f[:-1]) / dx
    if is_left:
        first = 2*df[0] - df[1]
        return np.concatenate([[first], df])
    else:
        last = 2*df[-1] + df[-2]
        return np.concatenate([df, [last]])

def rhs(is_left, rho, v, T):
    drhodx = diff(is_left, rho)
    dvdx = diff(is_left, v)
    dTdx = diff(is_left, T)
    drhodt = - drhodx * v - rho * dvdx - rho * v * d_ln_A_dx
    dvdt = - v * dvdx - 1 / gamma * (drhodx * T / rho + dTdx)
    dTdt =  - v * dTdx + (1 - gamma) * T * (dvdx + v * d_ln_A_dx)
    drhodt[0] = dTdt[0] = drhodt[-1] = 0.0
    return drhodt, dvdt, dTdt

def pred_step(old, diff):
    return old + dt * diff

def corr_step(old, pred, diff_pred, diff_corr):
    return (old + pred) / 2 + dt * (diff_pred + diff_corr) / 2

# time integration using mac cormack method
sol = []
every = 100
for step in range(nsteps):
    print(f"step = {step} / {nsteps}")
    # predictor step:
    drhodt_pred, dvdt_pred, dTdt_pred = rhs(step % 2 == 0, rho, v, T)
    rho_pred = pred_step(rho, drhodt_pred)
    v_pred = pred_step(v, dvdt_pred)
    T_pred = pred_step(T, dTdt_pred)
    # corrector step:
    drhodt_corr, dvdt_corr, dTdt_corr = rhs(step % 2 != 0, rho_pred, v_pred, T_pred)
    rho = corr_step(rho, rho_pred, drhodt_pred, drhodt_corr)
    v = corr_step(v, v_pred, dvdt_pred, dvdt_corr)
    T = corr_step(T, T_pred, dTdt_pred, dTdt_corr)
    if not np.all(np.isfinite(v)):
        break
    if step % every == 0:
        sol.append((rho, v, T))

fig, axs = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0)
fig.suptitle("t = 0.0")
rho, v, T = sol[0]
rho_line, = axs[0].plot(x, rho)
axs[0].set_ylabel("density rho")
v_line, = axs[1].plot(x, v)
axs[1].set_ylabel("velocity v")
T_line, = axs[2].plot(x, T)
axs[2].set_ylabel("temperature T")
axs[2].set_xlabel("x")

def animate(i):
    rho, v, T = sol[i]
    rho_line.set_ydata(rho)
    v_line.set_ydata(v)
    T_line.set_ydata(T)
    axs[0].set_ylim(np.min(rho), np.max(rho))
    axs[1].set_ylim(np.min(v), np.max(v))
    axs[2].set_ylim(np.min(T), np.max(T))
    fig.suptitle(f"time t = {dt * (i * every + 1):.2f}")

animation = a.FuncAnimation(fig, animate, interval=100, frames=len(sol))
plt.show()

