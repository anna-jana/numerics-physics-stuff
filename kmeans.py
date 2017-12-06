from __future__ import print_function, division

# setup
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
np.random.seed(0)

# parameters
num_clusters = 3
scale = 2
cluster_size = 30
steps = 7

# generate sample data
num_points = num_clusters*cluster_size

cluster_x = scale*np.random.rand(num_clusters)
cluster_y = scale*np.random.rand(num_clusters)

xs = cluster_x.repeat(cluster_size) + (np.random.rand(num_points)/4 - 0.5)
ys = cluster_y.repeat(cluster_size) + (np.random.rand(num_points)/4 - 0.5)

# Use k-means clustering to find the center of each cluster
# make inital guess x and y coordinates of the cluster centers
init_center_xs = scale*np.random.rand(num_clusters)
init_center_ys = scale*np.random.rand(num_clusters)

# now refine them
center_xs = init_center_xs
center_ys = init_center_ys

# Def. clostest point of a center: a point witch closest center is this one
for i in range(steps):
    # assign all points to there clostest center
    #                   points
    #                +----------->
    # cluster center | distance
    #                V
    dx = np.array([[(cx - x)**2 for x in xs] for cx in center_xs])
    dy = np.array([[(cy - y)**2 for y in ys] for cy in center_ys])
    # for the distance we would need a sqrt but since sqrt is monotonic we can omit it here
    # and get some performence gains
    dist = dx + dy

    # get all point closest to a center i
    min_center_index = np.argmin(dist, axis=0)

    # dont move centers to new locations if there have no clostest points
    bad_centers = set(range(num_clusters)) - set(min_center_index)

    # compute the center for each cluster
    center_xs = np.array([center_xs[i] if i in bad_centers else np.mean(xs[min_center_index == i])
        for i in range(num_clusters)])
    center_ys = np.array([center_ys[i] if i in bad_centers else np.mean(ys[min_center_index == i])
        for i in range(num_clusters)])

stop = time.time()
print("compute time: ", stop - start)

# plot it
colors = ["red", "blue", "green"]

# plot all clusters
for i in range(num_clusters):
    x = xs[i*cluster_size:(i + 1)*cluster_size]
    y = ys[i*cluster_size:(i + 1)*cluster_size]
    plt.scatter(x, y, color=colors[i], label="cluster " + str(i))

# plot the inital guesses and the final result
plt.scatter(init_center_xs, init_center_ys, color="black", label="inital guess")
plt.scatter(center_xs, center_ys, marker="x", color="black", label="final guess")

# labels
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("K Means Clustering")
plt.grid()
plt.show()

