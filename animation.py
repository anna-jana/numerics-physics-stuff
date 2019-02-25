import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def make_animation(x, ys, filename, fmt="x", init_plot=None):
    fig = plt.figure()
    ax = plt.axes(xlim=(np.min(x), np.max(x)), ylim=(np.min(ys), np.max(ys)))
    line, = ax.plot([], [], fmt)
    def init():
        if init_plot is not None:
            init_plot()
        line.set_data([], [])
        return line,
    def animate(i):
        print("step:", i)
        y = ys[i]
        line.set_data(x, y)
        return line,
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=ys.shape[0], interval=20, blit=True)
    anim.save(filename, fps=30, extra_args=["-vcodec", "libx264"])

if __name__ == "__main__":
    x = np.linspace(0, 8, 300)
    p = np.linspace(0, 30, 100)
    ys = np.sin(x[None, :]) - np.sin(p[:, None] * x[None, :])
    make_animation(x, ys, "anim.mp4", fmt="r-", init_plot=lambda: (plt.xlabel("x"), plt.ylabel("y")))
    plt.show()
