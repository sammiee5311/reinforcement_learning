import pygame
import collections
import numpy as np
from enum import Enum


Point = collections.namedtuple('Point', 'x, y')


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Snake:
    def __init__(self):
        # pygame.init()

        # GAME
        self.WIDTH = 300
        self.HEIGHT = 500
        # self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.score = 0

        # SNAKE
        self.snake_head = Point(self.WIDTH // 2,self.HEIGHT // 2)
        self.snake_body = collections.deque()
        self.snake_body.append(self.snake_head)
        self.direction = Direction.RIGHT

        # FRUIT
        self.fruit = Point((np.random.randint(10, self.WIDTH) // 10) * 10, (np.random.randint(10, self.HEIGHT) // 10) * 10)

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

        if 0 > x or self.WIDTH <= x or 0 > y or y >= self.HEIGHT or point in self.snake_body:
            return True
        return False

    # Display the game
    def display(self):
        self.SCREEN.fill(self.white)
        pygame.draw.rect(self.SCREEN, self.red, [self.fruit_x, self.fruit_y, 10, 10])

        for s in self.snake_body:
            x, y = s
            pygame.draw.rect(self.SCREEN, self.black, [x, y, 10, 10])

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
            self.move(10, 0)
        elif self.direction == Direction.LEFT:
            self.move(-10, 0)
        elif self.direction == Direction.UP:
            self.move(0, -10)
        else:
            self.move(0, 10)

    def move(self, x, y):
        self.snake_head = Point(self.snake_head.x + x, self.snake_head.y + y)
