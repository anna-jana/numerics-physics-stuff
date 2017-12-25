import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(42)

# generate test data
num_points = 12
num_clusters = 3
dim = 2
spread = 1.0
center_spread = 5.0

cluster_centers = np.random.uniform(-center_spread, center_spread, (num_clusters, dim))
cluster_of_points = np.random.randint(0, num_clusters, num_points)
cluster_centers_for_points = cluster_centers[cluster_of_points]
rel_coords = np.random.uniform(-spread, spread, (num_points, dim))
points = cluster_centers_for_points + rel_coords

# kmeans clustering
# find initial centers for the iteration
num_find_clusters = 3
steps = 10
num_points = points.shape[0]
upper = np.max(points, axis=0)
lower = np.min(points, axis=0)
initial_centers =  np.random.rand(num_find_clusters, dim) * (upper - lower) + lower

centers = initial_centers

for i in range(steps):
    # find closest center for each point
    centers_points_diff = points[None, :] - centers[:, None]
    centers_dists = np.linalg.norm(centers_points_diff, axis=2)
    closest_center_index = np.argmin(centers_dists, axis=0)

    # compute new centers
    new_centers = np.array([np.mean(points[closest_center_index == center_index], axis=0) for center_index in range(num_find_clusters)])
    centers = new_centers

final_centers = centers

# plt.plot(points[:,0], points[:,1], "xr", label="data")
plt.plot(cluster_centers[:,0], cluster_centers[:,1], "*k", label="real centers")
plt.plot(initial_centers[:, 0], initial_centers[:, 1], "*r", label="initial centers")
plt.plot(final_centers[:, 0], final_centers[:, 1], "*g", label="final centers")

for i in range(num_clusters): # clusters as generated
    cluster = points[cluster_of_points == i]
    plt.plot(cluster[:, 0], cluster[:, 1], "o", label="generated cluster %d" % i)

for i in range(num_find_clusters): # clusters as found
    cluster = points[closest_center_index == i]
    plt.plot(cluster[:, 0], cluster[:, 1], "x", label="infered cluster %d" % i)


plt.grid()
plt.title("kmeans clustering")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(numpoints=1)
plt.show()

