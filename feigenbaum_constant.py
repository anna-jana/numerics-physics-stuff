from pylab import *

# iterated map
def f(x, a): return a - x**2

# find point separated by eps
def extract_fixpoints(xs, eps=1e-2):
    fixpoints = []
    for x in xs:
        for fixpoint in fixpoints:
            if abs(x - fixpoint) < eps: # fixpoint already in list
                break
        else:
            fixpoints.append(x) # new fixpoint
    return fixpoints

# compute orbit diagram
num_to_collect = 100
num_transient = 1000 # this number has to be large
num_a = 600
x0 = 0.5 # exact value doesn't matter since we are transitioning into the attractor first
complete_x_list = []
complete_a_list = []
num_fixpoints = np.empty(num_a, dtype="int")
a_list = np.linspace(0.6, 1.5, num_a)

for k, a in enumerate(a_list):
    x = x0
    for i in range(num_transient): # transient
        x = f(x, a)

    # collect orbit
    x_list = [x]
    for i in range(num_to_collect - 1):
        x_list.append(f(x_list[-1], a))
    complete_a_list += [a] * num_to_collect
    complete_x_list += x_list

    num_fixpoints[k] = len(extract_fixpoints(x_list))

# compute the bifurcations and feigenbaum constant
l = np.log2(num_fixpoints)
li = l.astype("int")
valid = l == li # valid points are powers of two
# find the points where the count of fixpoints changes (the bifurcations)
# and one of them is valid (to avoid multiple bifurcations detected per bifurcation)
change = np.where(valid[:-1] & (num_fixpoints[:-1] != num_fixpoints[1:]))
# compute the a values for the bifurcations
a_change = a_list[change] + (a_list[1] - a_list[0]) / 2
# feigenbaum constant
feigenbaum_const = (a_change[1:-1] - a_change[:-2]) / (a_change[2:] - a_change[1:-1])

# plot
plt.subplot(2,1,1)
plt.plot(complete_a_list, complete_x_list, ".", ms=0.5, label="orbit")
for i, a in enumerate(a_change):
    plt.axvline(a, color="black", lw=0.5, label="bifurcations" if i == 0 else None)
plt.xticks([], [])
plt.ylabel("x")
plt.legend(framealpha=1.0)
plt.subplot(2,1,2)
plt.plot(a_list, num_fixpoints)
plt.xlabel("a")
plt.ylabel("count of fixpoints")
plt.suptitle(f"Feigenbaum constant: {feigenbaum_const}")
plt.tight_layout(h_pad=0.0)
plt.show()
