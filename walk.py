import numpy as np
import matplotlib.pyplot as plt

steps = 2000
do_walks = 400
end_points = np.empty(do_walks)

for i in range(do_walks):
    deltas = np.random.uniform(-1,1,steps)
    dist = np.cumsum(deltas)
    end_points[i] = dist[-1]

plt.subplot(2,1,1)
plt.hist(end_points, bins=31)
plt.xlabel("end point")
plt.ylabel("count")
plt.title("Random Walk after %i steps using %i walks" % (steps, do_walks))

plt.subplot(2,1,2)
plt.plot(dist)
plt.xlabel("step")
plt.ylabel("dist")
plt.title("Example Random Walk")

plt.tight_layout()
plt.show()
