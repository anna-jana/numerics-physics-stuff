from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

def find_clusters_from_centers(points, centers):
    # find closest center for each point
    num_centers = centers.shape[0]
    centers_points_diff = points[None, :] - centers[:, None]
    centers_dists = np.linalg.norm(centers_points_diff, axis=2)
    closest_center_index = np.argmin(centers_dists, axis=0)
    clusters = [points[closest_center_index == center_index] for center_index in range(num_centers)]
    return clusters

def kmeans_step(points, centers):
    # compute new centers
    new_centers = np.array([np.mean(cluster, axis=0) for cluster in find_clusters_from_centers(points, centers)])
    return new_centers

# generate test data
np.random.seed(42)
num_pts_per_cluster = 30
num_clusters = 3
span = 2.0
centers = span*np.random.rand(num_clusters, 2)
points = centers.repeat(num_pts_per_cluster, axis=0) + np.random.rand(num_pts_per_cluster * num_clusters, 2)

# kmeans clustering
# find initial centers for the iteration
num_find_clusters = num_clusters
steps = 10
initial_centers = np.random.uniform(np.min(points, axis=0), np.max(points, axis=0), size=(num_find_clusters, 2))

# iteration
center_steps = [initial_centers]
for i in range(steps):
    center_steps.append(kmeans_step(points, center_steps[-1]))
center_steps = np.array(center_steps)


# plotting
# plot the points of each cluster in a different color.
colors = ["red", "green", "blue"]
for color, cluster in zip(colors, find_clusters_from_centers(points, center_steps[-1])):
    plt.plot(cluster[:,0], cluster[:,1], "x", color=color)

# plot the path of the iteration of the cluster center
for i in range(num_find_clusters):
    plt.plot(center_steps[:, i, 0], center_steps[:, i, 1], "-k")

# plot the end centers
for i, color in enumerate(colors):
    x = center_steps[-1, i, 0]
    y = center_steps[-1, i, 1]
    plt.plot(x, y, "o", color=color)

plt.xlabel("x")
plt.ylabel("y")
plt.title("kmeans clustering")
plt.grid()

plt.show()
