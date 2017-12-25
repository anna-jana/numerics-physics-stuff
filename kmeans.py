from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(42)

# generate test data
num_points = 100
num_clusters = 3
dim = 2
spread = 1.0
center_spread = 3.0

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

colors = ["red", "green", "blue"]

# plt.plot(points[:,0], points[:,1], "xr", label="data")
plt.plot(cluster_centers[:,0], cluster_centers[:,1], "ok", label="real centers")
plt.plot(initial_centers[:, 0], initial_centers[:, 1], "or", label="initial centers")
plt.plot(final_centers[:, 0], final_centers[:, 1], "og", label="final centers")

for i, color in zip(range(num_clusters), colors): # clusters as generated
    cluster = points[cluster_of_points == i]
    plt.plot(cluster[:, 0], cluster[:, 1], "+", color=color, label="generated cluster %d" % i)

for i, color in zip(range(num_find_clusters), colors): # clusters as found
    cluster = points[closest_center_index == i]
    plt.plot(cluster[:, 0], cluster[:, 1], "x", color=color, label="infered cluster %d" % i)


plt.grid()
plt.title("kmeans clustering")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(np.min(points[:, 0]) - 1.0, np.max(points[:, 1]) + 4.0)
plt.legend(numpoints=1)
plt.show()

