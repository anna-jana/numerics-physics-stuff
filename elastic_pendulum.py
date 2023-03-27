# interactive simulation of elastic penduluum
import numpy as np
from scipy.integrate import solve_ivp
import pygame
from collections import deque

# parameter
window_width = 800 # pixels
window_height = 600 # pixels
aspect_ratio = window_width / window_height
window_size = np.array((window_width, window_height))
# physical parameters in simulation units
grav_accel = np.array((0.0, 1.0)) * 0.1
mass = 1.0
spring_constant = 1.0
spring_eqil_len = 0.2
# colors are RGB
background_color = 0x00_00_00
mount_color = 0x00_00_FF
mass_color = 0xFF_00_00
spring_color = 0xFF_FF_FF
mass_radius = int(window_width * 0.02) # pixel
mount_radius = int(window_width * 0.01) # pixel
num_spring_segments = 20
string_segment_length = 0.01 # simulation units
mount_pos = np.array((0.5, 0.1)) # simulation units
fps = 30
history_time = 20 # seconds
history_color = 0xFF_00_FF

# state and init
# simulation state
mass_pos = mass_vel = None # simulation units
def reset():
    global mass_pos, mass_vel
    mass_pos = np.array((0.7, 0.5))
    mass_vel = np.array((0.0, 0.0))
reset()

# interface state
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Elastic Pendulum Simulation")
running = True
clock = pygame.time.Clock()
advancing = False
modifing = False
offset = None
history = deque() # containts history of the trajectory in pixel positions
max_history_len = int(history_time * fps)

def line(a, b): # a and b in simulation units
    pygame.draw.line(window, spring_color,
            simulation_to_window_coords(a),
            simulation_to_window_coords(b))

def unpack(u):
    return u[:2], u[2:]

# in simulation units
def rhs(t, u):
    pos, vel = unpack(u)
    Delta = pos - mount_pos
    # fix that the coordinates x and y have different scales
    # window_width and window_height are different but our simulation coordinates are 0..1 for x and y each
    Delta[1] *= aspect_ratio
    d = np.linalg.norm(Delta)
    accel = - spring_constant / mass * (d - spring_eqil_len) * Delta / d + grav_accel
    return np.hstack([vel, accel])

# in simulation units
def get_mouse_pos():
    return np.array(pygame.mouse.get_pos()) / window_size

def simulation_to_window_coords(pos):
    return int(pos[0] * window_width), int(pos[1] * window_height)

# main loop
while running:
    # events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # keyboard
            if event.key == pygame.K_SPACE:
                advancing = not advancing
            elif event.key == pygame.K_r:
                reset()
                history.clear()
        elif not modifing and event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
            modifing = True
            offset = mass_pos - get_mouse_pos()
        elif modifing and event.type == pygame.MOUSEBUTTONUP and not pygame.mouse.get_pressed()[0]:
            mass_vel = np.array((0.0, 0.0))
            history.clear()
            modifing = False
        elif modifing and event.type == pygame.MOUSEMOTION:
            mass_pos = get_mouse_pos() + offset

    # draw
    window.fill(background_color)
    # history line
    if len(history) > 2:
        # dont connect the last and first point
        pygame.draw.lines(window, history_color, False, history)

    # draw spring
    diff = (mass_pos - mount_pos) / num_spring_segments
    delta = np.linalg.norm(diff)
    normal = np.array((diff[1], - diff[0])) / delta
    spring_width = (string_segment_length**2 - (delta / 4)**2)**0.5
    if np.isfinite(spring_width):
        for i in range(num_spring_segments):
            start = mount_pos + i * diff
            end = mount_pos + (i + 1) * diff
            left = start + 0.25 * diff + spring_width * normal
            right = start + 0.75 * diff - spring_width * normal
            line(start, left)
            line(left, right)
            line(right, end)
    else:
        # line is completly streched out
        line(mount_pos, mass_pos)

    # point masses
    pygame.draw.circle(window, mount_color,
                       simulation_to_window_coords(mount_pos),
                       mount_radius)
    pygame.draw.circle(window, mass_color,
                       simulation_to_window_coords(mass_pos),
                       mass_radius)

    pygame.display.flip()

    Delta_t = clock.tick(fps) / 1e3 # to seconds

    # advance
    if advancing and not modifing:
        # update history and limit to max_history_len points
        history.append(simulation_to_window_coords(mass_pos))
        if len(history) > max_history_len:
            history.popleft()
        # update simulation state
        u0 = np.hstack([mass_pos, mass_vel])
        sol = solve_ivp(rhs, (0, Delta_t), u0)
        final = sol.y[:, -1]
        mass_pos, mass_vel = unpack(final)
