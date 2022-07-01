import numpy as np, matplotlib.pyplot as plt

def compute_boxcounts(xs, epsilons):
    return np.array([len(set(tuple(np.floor(x / eps)) for x in xs)) for eps in epsilons])

# generate sierpinski triangle using the chaos game
def chaos_game(npoints):
    ps = list(map(np.array, [(0, 0), (1, 0), (0.5, np.sqrt(1 - 0.5**2))]))
    vs = [(0.5, 0.1)]
    for n in range(npoints):
        p = ps[np.random.randint(len(ps))]
        vs.append(0.5*(vs[-1] + p))
    return vs

np.random.seed(42)
npoints = 10000
neps = 100
points = np.array(chaos_game(npoints))
epsilons = np.geomspace(1 / npoints, 1, neps)
boxcounts = compute_boxcounts(points, epsilons)

x = np.log10(epsilons)
y = np.log10(boxcounts / npoints)
start_fit = -1.8
end_fit = -0.8
mask = np.where((x >= start_fit) & (x <= end_fit))[0]
slope, const = np.polyfit(x[mask], y[mask], 1)

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(points[:, 0], points[:, 1], ".", ms=1.0)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("sierpinski triangle from chaos game")
axs[1].plot(x, y, label="boxcounting")
axs[1].plot(x[mask], slope*x[mask] + const, label=f"fit $\\Delta = {-slope:.3f}$")
axs[1].axvline(start_fit, ls="--", color="k", label="fit range")
axs[1].axvline(end_fit, ls="--", color="k")
axs[1].set_xlabel(r"$\log_{10}(\varepsilon)$")
axs[1].set_ylabel(r"$\log_{10}(N_\mathrm{boxcounting} / N_\mathrm{samples})$")
axs[1].set_title(r"$N \sim \varepsilon^\Delta$")
axs[1].legend()
plt.suptitle("Box-Counting Dimension $\\Delta$")
plt.show()
