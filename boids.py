import numpy as np
import pygame

# parameter
window_width = 800 # pixels
window_height = 600 # pixels
aspect_ratio = window_width / window_height
window_size = np.array((window_width, window_height))
fps = 30
background_color = 0x00_00_00
boid_color = 0xFF_00_FF
boid_base = 0.01
boid_length = 0.02
boid_radius = int(0.02 * window_width)
num_boids = 50
private_space = 0.05
move_toward_center_amount = 0.01
avoid_collision_amount = 1.0
align_amount = 1/8

def simulation_to_window_coords(pos):
    return int(pos[0] * window_width), int(pos[1] * window_height)

def step(Delta_t, positions, velocities):
    idx = np.arange(positions.shape[0])
    # modify velocity accourding to the boid rules
    new_velocities = velocities.copy()
    for i in range(positions.shape[0]):
        # rule 1: move toward center
        center = np.mean(positions[idx != i], axis=0)
        new_velocities[i] += move_toward_center_amount * (center - positions[i])
        # rule 2: move away from neighboring boids to avoid collisions
        is_local_group = (np.linalg.norm(positions - positions[i]) <= private_space) & (idx != i)
        if not np.any(is_local_group):
            continue
        local_center = np.mean(positions[is_local_group], axis=0)
        new_velocities[i] += avoid_collision_amount * (positions[i] - local_center)
        # rule 3: align with velocity of other boids
        mean_velocity = np.mean(velocities[is_local_group], axis=0)
        new_velocities[i] += align_amount * mean_velocity
    # move boids
    new_positions = positions + Delta_t * new_velocities
    # limit position to be inside the simulation box
    for i in range(positions.shape[0]):
        for j in 0,1:
            if new_positions[i, j] < 0:
                new_positions[i, j] = 0
                new_velocities[i, j] *= -1
            max_dim = window_width if j == 0 else window_height
            if new_positions[i, j] > max_dim:
                new_positions[i, j] = max_dim
                new_velocities[i, j] *= -1
    return new_positions, new_velocities

def new():
    positions = np.random.uniform(0.25, 0.75, (num_boids, 2))
    alpha = np.random.uniform(0, 2*np.pi, num_boids)
    r = np.random.uniform(0, 0.1, num_boids)
    velocities = np.vstack([np.cos(alpha) * r, np.sin(alpha) * r]).T
    return positions, velocities

if __name__ == "__main__":
    positions, velocities = new()
    # interface state
    pygame.init()
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Boids")
    running = True
    advancing = False
    clock = pygame.time.Clock()
    # main loop
    while running:
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                advancing = not advancing
        # draw
        window.fill(background_color)
        for r, v in zip(positions, velocities):
            alpha = np.arctan2(v[1], v[0]) - np.pi / 2
            c = np.cos(alpha)
            s = np.sin(alpha)
            def transform(x, y):
                return (
                    int(window_width * (r[0] + c * x - s * y)),
                    int(window_width * (r[1] + s * x + c * y)),
                )
            triangle = [transform(- boid_base / 2, - boid_length / 2),
                        transform(+ boid_base / 2, - boid_length / 2),
                        transform(0, boid_length / 2)]
            pygame.draw.polygon(window, boid_color, triangle)
        pygame.display.flip()
        # advance
        Delta_t = clock.tick(fps) / 1e3 # to seconds
        if advancing:
            positions, velocities = step(Delta_t, positions, velocities)
