from __future__ import division, print_function
import math

import pygame

pygame.init()

window_size = window_width, window_height = 860, 460
rows, columns = 10, 10
fps = 20
black = (0, 0, 0)
blue = (0, 0, 255)
orange = (255, 165, 0)
white = (255, 255, 255)

std_color = black
mouse_over_color = blue
neighbor_color = orange
background_color = black
edge_color = white

class Hexagon(object):
    """
    A regular hexagon on the 2D plane witch has one side
    parallel to the X axis.
    Arguments:
        row: int - the row of the hexagon in the grid
        col: int - the column of the hexagon in the grid
    """

    def __init__(self, row, col, rows, cols, width=50):
        """
        Constructs a new hexagon.
        Arguments:
            row: int - the row of the hexagon in the grid
            col: int - the column of the hexagon in the grid
        """
        self.row, self.col = row, col

        self.width = width

        alpha = 2*math.pi/3
        self.a = width / (2*math.cos(alpha/2) + 1)
        self.h = math.cos(alpha/2)*self.a
        self.b = math.sin(alpha/2)*self.a*2

        if row % 2 == 0:
            self.x = self.a + self.h
        else:
            self.x = 0.0
        self.x += self.col*(self.a + width)
        self.y = (self.row + 1)*self.b/2
        self.index = (self.row, self.col)

        self.points = [
                (self.x, self.y),
                (self.x + self.h, self.y - self.b/2),
                (self.x + self.h + self.a, self.y - self.b/2),
                (self.x + width, self.y),
                (self.x + self.h + self.a, self.y + self.b/2),
                (self.x + self.h, self.y + self.b/2)
        ]

        rel_indices = [
                (1,0),
                (2,0),
                (-1,0),
                (-2,0),
        ]
        if self.row % 2 == 0:
            rel_indices += [(1, 1), (-1, 1)]
        else:
            rel_indices += [(-1, -1), (1, -1)]

        self.neighbor_indices = [((self.row + drow) % rows, (self.col + dcol) % cols)
                                 for drow, dcol in rel_indices]

    def draw(self, window, color):
        """
        Draws this hexagon on to this pygame surface using a
        griven color.
        Arguments:
            window - the pygame surface
            color - the color (as an RGB tuple)
        """
        pygame.draw.polygon(window, color, self.points)
        pygame.draw.lines(window, edge_color, True, self.points)

    def __contains__(self, point):
        """
        Tests if a given point is in this hexagon.
        Arguments:
            point - the point as an (x|y) tuple.
        """
        X, Y = point
        m1 = -self.b/(2*self.h)
        b1 = self.y - m1*self.x
        f1 = m1*X + b1
        m2 = -m1
        b2 = self.y - m2*self.x
        f2 = m2*X + b2
        b3 = self.y - m2*(self.x + self.width)
        f3 = m2*X + b3
        b4 = self.y - m1*(self.x + self.width)
        f4 = m1*X + b4
        return self.x + self.h <= X <= self.x + self.h + self.a and \
               self.y - self.b/2 <= Y <= self.y + self.b/2 or \
               self.x <= X <= self.x + self.h and \
               f1 <= Y <= f2 or \
               self.x + self.h + self.a <= X <= self.x + self.width and \
               f3 <= Y <= f4

grid = [Hexagon(row, col, rows, columns) for row in range(rows) for col in range(columns)]

window = pygame.display.set_mode(window_size)
pygame.display.set_caption("Hexgrid")
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.fill(background_color)

    for hexagon in grid:
        hexagon.draw(window, std_color)

    mouse_pos = pygame.mouse.get_pos()
    for hexagon in grid:
        if mouse_pos in hexagon:
            hexagon.draw(window, mouse_over_color)
            for other_hexagon in grid:
                if other_hexagon.index in hexagon.neighbor_indices:
                    other_hexagon.draw(window, neighbor_color)

    pygame.display.flip()
    clock.tick(fps)
