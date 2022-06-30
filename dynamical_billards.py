import numpy as np, matplotlib.pyplot as plt

def orthogonal(d):
    return  np.array([-d[1], d[0]])

safty_eps = 1e-15

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.normal = orthogonal(direction)
        self.d2 = np.linalg.norm(direction)**2

    def reflect(self, shape, x):
        # flipped = - self.direction
        # projected = n * (flipped @ n)
        # new_direction = flipped - 2 * (flipped - projected)
        n = shape.get_normal_at(x)
        n /= np.linalg.norm(n)
        projected = - n * (self.direction @ n)
        new_direction = self.direction + 2 * projected
        new_direction /= np.linalg.norm(new_direction) # HACK
        return Ray(x + safty_eps * new_direction, new_direction)
        # return Ray(x, new_direction)

class LineSegment:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        self.normal = orthogonal(self.direction)

    def intersect_with_ray(self, ray):
        # ray: x = o + t*d
        # line segment = (x - x0)*n = 0
        # (o + t*d - x0) *  n = 0
        # (o - x0) * n + t*d*n = 0
        # t = - (o - x0) * n / (d*n)
        A = ray.direction @ self.normal
        if A == 0.0:
            return None
        t = - ((ray.origin - self.start) @ self.normal) / A
        if t < 0.0:
            return None
        x = ray.origin + t * ray.direction
        s = np.linalg.norm(x - self.start) / self.length
        if 0.0 <= s <= 1.0:
            return x
        else:
            return None

    def get_normal_at(self, x):
        return self.normal

    def plot(self):
        plt.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], color="k")

class CircleSegment:
    def __init__(self, center, radius, arc_start, arc_end):
        self.center = np.array(center)
        self.radius = radius
        self.arc_start = arc_start
        self.arc_end = arc_end

    def intersect_with_ray(self, ray):
        # cirlce: |x - c| = r
        # ray: x = x0 + t*d
        # (x0 + t*d - c)^2 = r^2
        # (x0 - c)^2 + t^2*d^2 + 2*t*(x0 - c)*d - r^2 = 0
        # t^2 + 2*t*(x0 - c)*d / d^2 + (x0 - c)^2 / d^2  - r^2 / d^2 = 0
        p = 2*(ray.origin - self.center) @ ray.direction / ray.d2
        q = np.linalg.norm(ray.origin - self.center)**2 / ray.d2 - self.radius**2 / ray.d2
        disc = (p / 2)**2 - q
        if disc < 0.0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = -p/2 + sqrt_disc
        t2 = -p/2 - sqrt_disc
        if t1 < 0.0:
            return None # both t1 and t2 are negative
        def f(t):
            x = ray.origin + t * ray.direction
            diff = x - self.center
            angle = np.arctan2(diff[1], diff[0])
            if self.arc_start <= np.pi + angle <= self.arc_end:
                return x
            else:
                return None
        if t2 < 0.0:
            return f(t1) # only t1 is positive
        else:
            ans = f(t2) # both are positive
            if ans is None:
                return f(t1)
            else:
                return ans

    def get_normal_at(self, x):
        return self.center - x

    def plot(self):
        angle = np.linspace(self.arc_start, self.arc_end, 200)
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        plt.plot(x, y, color="k")

def step_dynamical_billards(geometry, state):
    intersections = [shape.intersect_with_ray(state) for shape in geometry]
    x0 = state.origin
    def min_key_fn(i):
        intersection = intersections[i]
        if intersection is None: return np.inf
        d = np.linalg.norm(intersection - x0)
        if d < safty_eps: return np.inf
        return d
    intersection_index = min(range(len(intersections)), key=min_key_fn)
    assert intersections[intersection_index] is not None, "the ray escaped"
    new_state = state.reflect(geometry[intersection_index], intersections[intersection_index])
    return new_state

def simulate_dynamical_billards(geometry, start_points, start_angles, nsteps):
    history = []
    states = [Ray(np.array(start_point), np.array([np.cos(start_angle), np.sin(start_angle)]))
            for start_point, start_angle in zip(start_points, start_angles)]
    for step in range(nsteps):
        history.append(states)
        states = [step_dynamical_billards(geometry, state) for state in states]
    history.append(states)
    return history

def plot_dynamical_billards(geometry, history, do_show_directions):
    for shape in geometry:
        shape.plot()
    xs = np.array([[s.origin for s in states] for states in history]) # shape = (nsteps, nparticles, 2)
    for particle_index in range(xs.shape[1]):
        plt.plot(xs[:, particle_index, 0], xs[:, particle_index, 1])
    if do_show_directions:
        vs = np.array([[s.direction for s in states] for states in history]) # shape = (nsteps, nparticles, 2)
        for particle_index in range(xs.shape[1]):
            plt.quiver(xs[:, particle_index, 0], xs[:, particle_index, 1],
                       vs[:, particle_index, 0], vs[:, particle_index, 1])

def run_dynamical_billards(geometry, start_points, start_angles, nsteps, do_show_directions=True):
    history = simulate_dynamical_billards(geometry, start_points, start_angles, nsteps)
    plot_dynamical_billards(geometry, history, do_show_directions)
    plt.show()

box = [LineSegment((-1.0, +1), (+1, +1)), LineSegment((+1.0, +1), (+1, -1)),
       LineSegment((+1.0, -1), (-1, -1)), LineSegment((-1.0, -1), (-1, +1))]
circle = [CircleSegment((0.0, 0), 0.5, 0.0, 2*np.pi)]
sinai = box + circle

# run_dynamical_billards(box, [(0.0, 0)], [np.pi / 2 - 0.1], 20)
# run_dynamical_billards(circle, [(0.25, 0)], [np.pi / 2 - 0.1], 10)
# run_dynamical_billards(sinai, [(0.75, 0)], [np.pi / 2 - 0.1], 10)
nparticles = 10
eps = 1e-8
nsteps = 23
run_dynamical_billards(sinai, [(0.75, 0)]*nparticles,
        np.pi / 2 + 0.1 + np.linspace(-eps, eps, nparticles), nsteps, do_show_directions=False)
