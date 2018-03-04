import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant as G
from scipy.integrate import odeint

def star_positions_at(time, omega, r):
    angle = omega*time
    x1 = np.array([r*np.cos(angle), r*np.sin(angle)])
    second_angle = angle + np.pi # on the otherside
    x2 = np.array([r*np.cos(second_angle), r*np.sin(second_angle)])
    return x1, x2

def acceleration_from(x, star_x, star_mass):
    to_star = star_x - x
    d = np.linalg.norm(to_star)
    if d == 0.0:
        raise Exception("singularity")
    return G*star_mass/d**3*to_star

def simulate(seed=0, years=1.0, h=6300.0, x0=None, v0=None,
        M=2e30, stars_distance=150e6*1000.0, do_plot=True):
    """
    Run a simulation of a restricted tree body problem.
    seed = 0 # seed for the random number generator
    years = 1.0 # number of years to simulate WARNING: output time is in seconds!!!!
    steps = 5000 # steps to compute
    x0 = None # initial position of the body [m]
    v0 = none # initial velocity of the body [m/s]
    M = 2e40 # mass of each sun [kg]
    stars_distance = 150e6*1000.0 # distance between the two stars in [m]
    do_plot = True # plot the result?
    """
    np.random.seed(seed)

    # other stars parameter
    stars_radius = stars_distance/2.0 # [m]
    stars_speed = np.sqrt(G/stars_distance) # [m/s]
    stars_orbit_perimeter = np.pi*stars_distance # [m]
    stars_angular_velocity = 2*np.pi*stars_speed/stars_orbit_perimeter # radians per second [1/s]
    T_star = stars_orbit_perimeter/stars_speed/(365*24*60*60) # [yr]

    # make simulation time points
    simulation_time = years*365*24*60*60.0 # [s]
    ts = np.arange(0, simulation_time, h)

    # random starting position and velocity
    if x0 is None:
        x0 = np.random.rand(2)*2*stars_distance - stars_distance # [m]
    if v0 is None:
        v0_angle = np.random.uniform(0., 2*np.pi) # radiant [1]
        min_speed = 1000*5.0 # [m/s]
        max_speed = 1000*6.0 # [m/s]
        v0_mag = np.random.uniform(min_speed, max_speed) # [m/s]
        v0 = v0_mag*np.array([np.cos(v0_angle), np.sin(v0_angle)]) # [m/s]
    y0 = np.zeros(4)
    y0[:2] = x0
    y0[2:] = v0

    # make a rhs function
    def rhs(y, t):
        x, v = y[:2], y[2:]
        star_x1, star_x2 = star_positions_at(t, stars_angular_velocity, stars_radius)
        a = acceleration_from(x, star_x1, M) + acceleration_from(x, star_x2, M)
        res = np.zeros(4)
        res[:2] = v
        res[2:] = a
        return res

    # solve the system
    res = odeint(rhs, y0, ts)

    # plot the orbit
    if do_plot:
        xs = res[:,0]
        ys = res[:,1]
        plt.plot(xs, ys, label="orbit of the body")
        plt.scatter([x0[0]],[x0[1]], label="starting point")
        a = np.linspace(0, 2*np.pi, 1000)
        star_x = stars_radius*np.cos(a)
        star_y = stars_radius*np.sin(a)
        plt.plot(star_x, star_y, label="orbit of the two stars")
        plt.legend()
        title_format = """
        Restricted tree body problem
        M_star = {0}[kg], d_star = {1}[m]
        x0 = {2}[m]
        v0 = {3}[m/s]
        T_star = {4}[years]
        """
        plt.title(title_format.format(M, stars_distance, x0, v0, T_star))
        plt.xlabel("x")
        plt.ylabel("y")

        plt.show()

    return res
