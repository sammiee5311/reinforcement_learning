import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import collections
import pygame

EPISODES = 100000
COLLISION_PENALTY = 850
POINT_REWARD = 50
ALIVE_REWARD = 1
epsilon = 0.9
EPS_DECAY = 0.99998
SHOW_EVERY = 1000

LEARNING_RATE = 0.3
DISCOUNT = 0.95
STEP = 1000


class Snake:
    def __init__(self):
        # pygame.init()

        # GAME
        self.WIDTH = 300
        self.HEIGHT = 500
        # self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.score = 0

        # SNAKE
        self.snake_x = self.WIDTH // 2
        self.snake_y = self.HEIGHT // 2
        self.snake_body = collections.deque([(self.snake_x, self.snake_y)])

        # FRUIT
        self.fruit_x = (np.random.randint(10, self.WIDTH) // 10) * 10
        self.fruit_y = (np.random.randint(10, self.HEIGHT) // 10) * 10

    def __sub__(self, other):
        x = self.snake_x - other.x
        y = self.snake_y - other.y
        return (x, y)

    # Check if the coordinates are in bound.
    def is_collision(self):
        if 0 < self.snake_x <= self.WIDTH and 0 < self.snake_y <= self.HEIGHT or (
        self.snake_x, self.snake_y) in self.snake_body:
            return False
        return True

    def action(self, direction):
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

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

for x in range(-300, 301):
    for y in range(-500, 501):
        q_table[(x, y)] = [0 for _ in range(4)]

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print("# : %d, epsilon : %f" % (episode, epsilon))
        print(f'{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}')

    snake = Snake()
    episode_reward = 0

    for _ in range(STEP):
        reward = 0
        new_x = snake.snake_x - snake.fruit_x
        new_y = snake.snake_y - snake.fruit_y
        obs = (new_x, new_y)

        if np.random.random() > epsilon:
            choice = np.argmax(q_table[obs])
        else:
            choice = np.random.randint(0, 4)

        snake.action(choice)

        collision = snake.is_collision()

        snake.snake_body.popleft()
        snake.snake_body.append((snake.snake_x, snake.snake_y))

        if collision:
            reward = -COLLISION_PENALTY
        else:
            if snake.snake_x == snake.fruit_x and snake.snake_y == snake.fruit_y:
                while snake.snake_x == snake.fruit_x and snake.snake_y == snake.fruit_y:
                    snake.fruit_x = (np.random.randint(10, snake.WIDTH) // 10) * 10
                    snake.fruit_y = (np.random.randint(10, snake.HEIGHT) // 10) * 10
                snake.snake_body.append((snake.snake_x, snake.snake_y))
                reward = POINT_REWARD
            reward += ALIVE_REWARD

        new_x = snake.snake_x - snake.fruit_x
        new_y = snake.snake_y - snake.fruit_y
        new_obs = (new_x, new_y)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][choice]

        if reward == -COLLISION_PENALTY:
            new_q = -COLLISION_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][choice] = new_q
        episode_reward += reward

        if collision:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg, color='indigo')
plt.ylabel("reward on %d" % SHOW_EVERY)
plt.xlabel("episode #")
plt.show()

with open("q_table.pickle", "wb") as file:
    pickle.dump(q_table, file)