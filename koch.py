import numpy as np
import matplotlib.pyplot as plt

initial_points = [np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(1**2 - (0.5)**2)])]

def add_iteration(ps):
    new = []
    for p1, p5 in zip(ps, ps[1:] + [ps[0]]):
        between = p5 - p1
        p2 = p1 + between/3.0
        p4 = p1 + 2.0*between/3.0
        l = np.linalg.norm(between)
        ln = np.sqrt((l/3.0)**2 - (l/6)**2)
        off_normal = np.array([between[1], -between[0]])/l
        off = off_normal*ln
        p3 = p1 + between/2.0 + off
        new.append(p1)
        new.append(p2)
        new.append(p3)
        new.append(p4)
    return new

def koch(n):
    points = initial_points
    for i in range(n):
        points = add_iteration(points)
    return points

def plot_koch(n):
    ps = koch(n)
    xs = [p[0] for p in ps]
    xs.append(ps[0][0])
    ys = [p[1] for p in ps]
    ys.append(ps[0][1])
    plt.plot(xs,ys)

if __name__ == "__main__":
    plot_koch(6)
    plt.title("Koch Curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
