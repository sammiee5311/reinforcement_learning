import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from matplotlib import style
import collections
import pygame

style.use('ggplot')

EPISODES = 100000
COLLISION_PENALTY = -500
POINT_REWARD = 50
ALIVE_REWARD = 1
epsilon = 0.9
EPS_DECAY = 0.99998
SHOW_EVERY = 5000

LEARNING_RATE = 0.3
DISCOUNT = 0.95
STEP = 500


class Snake:
    def __init__(self):
        pygame.init()

        # GAME
        self.WIDTH = 300
        self.HEIGHT = 500
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.score = 0
        self.game_over = False

        # SNAKE
        self.snake_x = self.WIDTH // 2
        self.snake_y = self.HEIGHT // 2
        self.snake_body = collections.deque([(self.snake_x, self.snake_y)])
        self.n_tail = 0

        # FRUIT
        self.fruit_x = (np.random.randint(1, self.WIDTH) // 10) * 10
        self.fruit_y = (np.random.randint(1, self.HEIGHT) // 10) * 10

    def choice(self, direction):
        if direction == 0:
            self.move(-10, 0)
        elif direction == 1:
            self.move(10, 0)
        elif direction == 2:
            self.move(0, -10)
        else:
            self.move(0, 10)

    def move(self, x, y):
        self.snake_x += x
        self.snake_y += y


q_table = collections.defaultdict(list)

episode_rewards = []

snake = Snake()

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

choice = 0

while not snake.game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                choice = 0
            elif event.key == pygame.K_RIGHT:
                choice = 1
            elif event.key == pygame.K_UP:
                choice = 2
            elif event.key == pygame.K_DOWN:
                choice = 3

    snake.choice(choice)

    if snake.snake_x >= snake.WIDTH or snake.snake_x < 0 or snake.snake_y >= snake.HEIGHT or snake.snake_y < 0 or (snake.snake_x,snake.snake_y) in snake.snake_body:
        snake.game_over = True

    snake.snake_body.popleft()
    snake.snake_body.append((snake.snake_x, snake.snake_y))
    snake.SCREEN.fill(white)

    if snake.snake_x == snake.fruit_x and snake.snake_y == snake.fruit_y:
        while snake.snake_x == snake.fruit_x and snake.snake_y == snake.fruit_y:
            snake.fruit_x = (np.random.randint(1, snake.WIDTH) // 10) * 10
            snake.fruit_y = (np.random.randint(1, snake.HEIGHT) // 10) * 10
        snake.n_tail += 1
        snake.snake_body.append((snake.snake_x, snake.snake_y))

    pygame.draw.rect(snake.SCREEN, red, [snake.fruit_x, snake.fruit_y, 10, 10])

    for s in snake.snake_body:
      x, y = s
      pygame.draw.rect(snake.SCREEN, black, [x, y, 10, 10])

    time.sleep(0.09)
    pygame.display.update()

pygame.quit()