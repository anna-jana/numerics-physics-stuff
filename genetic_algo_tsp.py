import functools
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2023)

def random_solutions(npoints, nsolutions):
    solutions = np.arange(npoints)[None, :].repeat(nsolutions, axis=0)
    for i in range(nsolutions):
        np.random.shuffle(solutions[i])
    return solutions

def compute_score(xs, sol):
    ps = xs[sol]
    return np.sum(np.linalg.norm(ps[:-1] - ps[1:], axis=0))

def mutate(sol, nswaps):
    for _ in range(nswaps):
        i = np.random.randint(len(sol))
        j = np.random.randint(len(sol))
        sol[i], sol[j] = sol[j], sol[i]

# parameter
npoints = 20
nsolutions = 40
dim = 2
keep_percent = 0.5
nswaps = 2
nsteps = 1000

# generate data
xs = np.random.rand(npoints, dim)

# init
to_keep = int(keep_percent*nsolutions)
nnew = nsolutions - to_keep
solutions = random_solutions(npoints, nsolutions)
best_score = []

for i in range(nsteps):
    # score solutions
    scores = np.array([compute_score(xs, solutions[i])
                       for i in range(nsolutions)])
    # find the best
    best = np.argsort(scores)
    # keep track of the best solution
    best_score.append(scores[best[0]])
    # best solutions so far
    keep = best[:to_keep]
    # create mutations
    mutated = solutions[np.random.choice(keep, nnew)]
    for i in range(nnew):
        mutate(mutated[i], nswaps)
    # combine the best solutions so far and their mutations
    solutions = np.vstack([solutions[keep], mutated])

best_solution = min(solutions, key=functools.partial(compute_score, xs))

# plotting
plt.figure(layout="constrained")
plt.subplot(1,2,1)
plt.plot(best_score)
plt.xlabel("generation")
plt.ylabel("score (total length of path)")
plt.title("evolution of the best solution's score")
plt.subplot(1,2,2)
ps = xs[best_solution]
plt.plot(ps[:, 0], ps[:, 1], "-")
plt.plot(xs[:, 0], xs[:, 1], "o")
plt.axis("off")
plt.title("best solution found")
plt.suptitle("Traveling Salesperson Problem with Genetic Algorithm")
plt.show()
