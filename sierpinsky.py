from __future__ import division

import numpy as np
import pygame
import matplotlib.pyplot as plt

pygame.init()

def sierpinsky(x=0, y=0, l=10, depth=8):
    if depth == 0:
        return []
    h = np.sqrt(2)/3.0*l
    ps = [(x - l/2, y + h), (x, y), (x + l/2, y + h)]
    ls = list(zip(ps, ps[1:] + [ps[0]]))
    ls += sierpinsky(x, y, l/2, depth - 1)
    ls += sierpinsky(x - l/4, y + h/2, l/2, depth - 1)
    ls += sierpinsky(x + l/4, y + h/2, l/2, depth - 1)
    return ls


def view(lines):
    xs = [l[0][0] for l in lines] + [l[1][0] for l in lines]
    ys = [l[0][1] for l in lines] + [l[1][1] for l in lines]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    window_width = 600
    window_height = int(window_width*(max_y - min_x)/(max_x - min_x))
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Sierpinsky Triangle")

    white = pygame.Color(255, 255, 255)
    black = pygame.Color(0, 0, 0)
    background_color = white
    line_color = black

    window.fill(background_color)

    def map_range(x, min_x, max_x, min_y, max_y):
        return (x - min_x)/(max_x - min_x)*(max_y - min_y) + min_y

    def map_x(x):
        return map_range(x, min_x, max_x, 0, window_width)

    def map_y(y):
        return map_range(y, min_y, max_y, 0, window_height)

    for p1, p2 in lines:
        p1 = map_x(p1[0]), map_y(p1[1])
        p2 = map_x(p2[0]), map_y(p2[1])
        pygame.draw.line(window, line_color, p1, p2)

    pygame.display.update()

    # keep the window open until someone presses quit
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.delay(500)

if __name__ == "__main__":
    view(sierpinsky())
