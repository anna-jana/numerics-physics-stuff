from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

stepper_xor_wave = lambda p, c: p ^ np.roll(c, -1) ^ np.roll(c, +1)
stepper_xor_klein_goron = lambda p, c: p ^ c ^ np.roll(c, -1) ^ np.roll(c, +1)
stepper_xor_full = lambda p, c: reduce(lambda x, y: x ^ y, [np.roll(s, d) for s in [p, c] for d in [-1,0,1]])

def sim(stepper, size=300, steps=300, title=""):
    rand_bits = lambda: np.random.randint(0, 2, size)
    p, c = rand_bits(), rand_bits()
    history = [p]
    for i in range(steps):
        p, c = c, stepper(p, c)
        history.append(p)
    plt.figure(); plt.pcolormesh(history); plt.title(title)

sim(stepper_xor_wave, title="xor wave")
sim(stepper_xor_klein_goron, title="xor klein gordon")
sim(stepper_xor_full, title="xor full?")
plt.show()

