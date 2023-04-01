import matplotlib.pyplot as plt
import numpy as np

def run_L_system(nsteps, start, rules):
    state = start
    for i in range(nsteps):
        state = "".join(symbol if symbol not in rules else rules[symbol]
                        for symbol in state)
    return state

def draw_L_system(steps, angle):
    turn_left = np.exp(1j * angle)
    turn_right = np.conj(turn_left)
    stack = []
    current_direction = 1j
    current_position = 0
    for step in steps:
        if step in "FG":
            new_position = current_position + current_direction
            plt.plot([current_position.real, new_position.real],
                     [current_position.imag, new_position.imag],
                     color="green")
            current_position = new_position
        elif step == "+":
            current_direction *= turn_left
        elif step == "-":
            current_direction *= turn_right
        elif step == "[":
            stack.append((current_position, current_direction))
        elif step == "]":
            current_position, current_direction = stack.pop()

rules = {
    "F": "FF",
    "X": "F+[[X]-X]-F[-FX]+X",
}
start = "X"
nsteps = 6
angle = np.deg2rad(25)
steps = run_L_system(nsteps, start, rules)
plt.figure()
draw_L_system(steps, angle)
ax = plt.gca()
ax.set_aspect("equal")
plt.title("\"Barnsley fern\" like plant")
plt.axis("off")
plt.show()


