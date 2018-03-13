from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

def kmeans_step(points, centers):
    # find closest center for each point
    centers_points_diff = points[None, :] - centers[:, None]
    centers_dists = np.linalg.norm(centers_points_diff, axis=2)
    closest_center_index = np.argmin(centers_dists, axis=0)

    # compute new centers
    new_centers = np.array([np.mean(points[closest_center_index == center_index], axis=0) for center_index in range(num_find_clusters)])

    return new_centers

# generate test data
np.random.seed(42)
num_pts_per_cluster = 10
num_clusters = 3
centers = 5*np.random.rand(num_clusters, 2)
points = centers.repeat(num_pts_per_cluster, axis=0) + np.random.rand(num_pts_per_cluster * num_clusters, 2)

# kmeans clustering
# find initial centers for the iteration
num_find_clusters = num_clusters
steps = 10
initial_centers = np.random.uniform(np.min(points, axis=0), np.max(points, axis=0), size=(num_find_clusters, 2))

center_steps = [initial_centers]
for i in range(steps):
    center_steps.append(kmeans_step(points, center_steps[-1]))
center_steps = np.array(center_steps)

plt.plot(points[:, 0], points[:, 1], "xk")

for i in range(num_find_clusters):
    plt.plot(center_steps[:, i, 0], center_steps[:, i, 1], "-k")

plt.xlabel("x")
plt.ylabel("y")
plt.title("kmeans clustering")
plt.grid()

plt.show()
