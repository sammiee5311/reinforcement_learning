import pygame
import collections
import numpy as np
import random
from enum import Enum


Point = collections.namedtuple('Point', 'x, y')
BLOCK = 20


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Snake:
    def __init__(self):
        pygame.init()

        # GAME
        self.WIDTH = 640
        self.HEIGHT = 480
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.score = 0

        # SNAKE
        self.snake_head = Point(self.WIDTH / 2,self.HEIGHT / 2)
        self.snake_body = [self.snake_head]
        # self.snake_body.append(self.snake_head)
        self.direction = Direction.RIGHT

        # FRUIT
        self.fruit = Point((random.randint(0, self.WIDTH-BLOCK) // BLOCK) * BLOCK, (random.randint(0, self.HEIGHT-BLOCK) // BLOCK) * BLOCK)

        # COLOR
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)

    # Check if the coordinates are in bound.
    def is_collision(self, point):
        if not point:
            x, y = self.snake_head
        else:
            x, y = point.x, point.y

        if 0 > x or self.WIDTH-BLOCK < x or 0 > y or y > self.HEIGHT-BLOCK or point in self.snake_body:
            return True
        return False

    # Display the game
    def display(self):
        self.SCREEN.fill(self.white)
        pygame.draw.rect(self.SCREEN, self.red, [self.fruit.x, self.fruit.y, BLOCK, BLOCK])

        for s in self.snake_body:
            x, y = s
            pygame.draw.rect(self.SCREEN, self.black, [x, y, BLOCK, BLOCK])

        pygame.display.update()

    def action(self, direction):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(direction, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(direction, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        if self.direction == Direction.RIGHT:
            self.move(BLOCK, 0)
        elif self.direction == Direction.LEFT:
            self.move(-BLOCK, 0)
        elif self.direction == Direction.UP:
            self.move(0, -BLOCK)
        else:
            self.move(0, BLOCK)

    def move(self, x, y):
        self.snake_head = Point(self.snake_head.x + x, self.snake_head.y + y)
