"""
Utils for drawing ODE stuff.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def plot_direction_field(f, x_range, y_range,
                ode_desc=None, xaxis="x", yaxis="y",
                res=20, x_res=None, y_res=None,
                arrow_len=1.0,
                integral_curves_at=[], curves_res=1000):
    """
    Plot a dircetion field of a first oder ODE y' = f(x, y) and
    some integral curves in that dircetion field.
    Arguments:
        f: Some function witch computes the right hand side of the ODE.
        x_range: The range of x values to plot (x_min, x_max).
        y_range: The range of y values to plot (y_min, y_max).
    Options:
        ode_desc=None: A textual description of the ode (e.g. a formula or a name).
                  (if ode_desc is None, no title is set)
        xaxis="x": Name of the x axis.
        yaxis"y": Name of the y axis.
        res=20: Number of arrows to draw per axis.
                res is ignored if a resultion of a specific axis is set.
        arrow_len=1.0: Length of the arrows to draw.
        integral_curves_at=[]: A list of (x_0, y_0) tuples as starting points
                            for the integral curves.
                            The integral curves are drawn both forward and backward in x.
        curves_res=1000: Number of points to use for the computation of the integral curves.
                         (for forward and backward each)
    """
    # set up all the labels
    if ode_desc is not None:
        plt.title("Direction field of {}".format(ode_desc))

    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

    x_res = res if x_res is None else x_res
    y_res = res if y_res is None else y_res

    # plot the dircetion field
    x = np.linspace(x_range[0], x_range[1], x_res)
    y = np.linspace(y_range[0], y_range[1], y_res)
    X, Y = np.meshgrid(x, y)

    F = f(X, Y)

    arrow_angle = np.arctan(F)
    arrow_x = arrow_len * np.cos(arrow_angle)
    arrow_y = arrow_len * np.sin(arrow_angle)

    plt.quiver(X, Y, arrow_x, arrow_y)

    # plot all integral curves
    g = lambda y, x: f(x, y)

    plt.ylim(y_range)

    for (x0, y0) in integral_curves_at:
        # backward in x
        xs_backward = np.linspace(x0, x_range[0], curves_res)
        ys_backward = odeint(g, y0, xs_backward)[:,0]
        plt.plot(xs_backward, ys_backward, 'r')

        # forward in x
        xs_forward = np.linspace(x0, x_range[1], curves_res)
        ys_forward = odeint(g, y0, xs_forward)[:,0]
        plt.plot(xs_forward, ys_forward, 'r')


def example1():
    plot_direction_field(lambda x, y: y, (-2, 2), (-2, 2), integral_curves_at=[(0, 1), (1., 1), (0., 0.)], ode_desc="y' = y")

def example2():
    plot_direction_field(lambda x, y: -x/y, (-2,2), (-2,2), integral_curves_at=[(1.,1.)], ode_desc="y' = -x/y")

def example3():
    plot_direction_field(lambda x, y: 1 + x - y, (-2, 2), (-2, 2), integral_curves_at=[(-1.5, 1.0)], ode_desc="y' = 1 + x - y")
