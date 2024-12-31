# http://persson.berkeley.edu/distmesh/persson04mesh.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# signed distance function
def circle(x, y, r):
    return x**2 + y**2 - r**2

def difference(s1, s2):
    return np.maximum(s1, -s2)

############################# parameter ###############################
# input for algorithm
def sdf(x, y):
    return difference(circle(x, y, 0.4), circle(x, y, 0.2))

def element_size_function(x, y):
    return 1 / np.sqrt(x**2 + y**2)

bounding_box = ((-0.5, 0.5), (-0.5, 0.5))

# convergence parameters
h0 = 0.05
dt = 0.2
fscale = 1.2
k = 1.0
conv_tol = 1e-3
retri_tol = 1e-1
geo_tol = 1e-3 * h0 # geps
ndim = 2

diff_eps = np.cbrt(np.finfo(np.float64).eps) * h0
np.random.seed(32123132)


############################ algorithm ################################
# uniform points in bounding box
xx, yy = np.meshgrid(np.arange(bounding_box[0][0], bounding_box[0][1], h0),
                     np.arange(bounding_box[1][0], bounding_box[1][1], h0))
points = np.vstack([xx.ravel(), yy.ravel()]).T

# remove points outside of domain
points = points[sdf(points[:, 0], points[:, 1]) < geo_tol, :]
npoints = points.shape[0]
# reject with density
prop = 1 / element_size_function(points[:, 0], points[:, 1])**2
prop /= np.max(prop)
points = points[np.random.rand(npoints) < prop, :]
npoints = points.shape[0]

def triangulate_points(points):
    delaunay = Delaunay(points)
    centroids = (points[delaunay.simplices[:, 0]] + points[delaunay.simplices[:, 1]] + points[delaunay.simplices[:, 2]]) / 3.0
    is_inside_domain = sdf(centroids[:, 0], centroids[:, 1]) < geo_tol
    domain_simplicies = delaunay.simplices[is_inside_domain, :]
    simplex_connections = np.roll(domain_simplicies.repeat(2, axis=1), 1, axis=1).reshape(-1, 2)
    connections = np.unique(np.sort(simplex_connections, axis=1), axis=0)
    return domain_simplicies, connections

old_points = np.inf
steps = 0
while True:
    retriangulate_change = np.max(np.linalg.norm(old_points - points, axis=1))
    if retriangulate_change > retri_tol:
        # retriangulate
        old_points = points.copy()
        _, connections = triangulate_points(points)

    # compute F
    mid_points = (points[connections[:, 0]] + points[connections[:, 1]]) / 2.0
    d = points[connections[:, 0]] - points[connections[:, 1]]
    l = np.linalg.norm(d, axis=1)
    element_size = element_size_function(mid_points[:, 0], mid_points[:, 1])
    scaling_factor = np.sqrt(np.sum(l**2) / np.sum(element_size**2))
    l0 = fscale * scaling_factor * element_size
    #fs = np.maximum((k * (l0 - l) / l)[:, None] * d, 0.0)
    fs = np.where(l < l0, k * (l0 - l) / l, 0.0)[:, None] * d

    F = np.zeros_like(points)
    for (i, j), f in zip(connections, fs):
        F[i, :] += f
        F[j, :] -= f

    # forward euler iteration
    points += dt * F

    # move points back into interior
    while True:
        s = sdf(points[:, 0], points[:, 1])
        bad = np.sum(s > geo_tol)
        print("points outside of domain:", bad)
        if bad == 0:
            break
        is_outside = s > 0.0
        outside_points = points[is_outside, :]
        gradx = (sdf(outside_points[:, 0] + diff_eps, outside_points[:, 1]) -
                sdf(outside_points[:, 0] - diff_eps, outside_points[:, 1])) / (2*diff_eps)
        grady = (sdf(outside_points[:, 0], outside_points[:, 1] + diff_eps) -
                sdf(outside_points[:, 0], outside_points[:, 1] - diff_eps)) / (2*diff_eps)
        points[is_outside, 0] -= s[is_outside] * gradx
        points[is_outside, 1] -= s[is_outside] * grady

    # check if the force is vanishing on each object
    is_interior = sdf(points[:, 0], points[:, 1]) < - geo_tol
    movement_change = np.max(np.sqrt(dt) * np.linalg.norm(F[is_interior, :], axis=1)) / h0
    print("change:", movement_change)
    if movement_change < conv_tol:
        break

domain_simplicies, connections = triangulate_points(points)

# visualization
plt.figure()
x = np.linspace(-0.5, 0.5, 50)
x, y = np.meshgrid(x, x)
plt.pcolormesh(x, y, np.where(sdf(x, y) < 0.0, 1.0, 0.0))
plt.triplot(points[:, 0], points[:, 1], domain_simplicies)
plt.scatter(points[:, 0], points[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
