import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import collections
import pygame

style.use('ggplot')

EPISODES = 100000
COLLISION_PENALTY = 850
POINT_REWARD = 50
ALIVE_REWARD = 1
epsilon = 0.9
EPS_DECAY = 0.99998
SHOW_EVERY = 5000

LEARNING_RATE = 0.3
DISCOUNT = 0.95
STEP = 500


class game:
    def __init__(self):
        pygame.init()

        self.bird_x = 50
        self.bird_y = 300

        self.OBSTACLE_WIDTH = 70
        self.OBSTACLE_HEIGHT = np.random.randint(200, 400)
        self.OBSTACLE_COLOR = (211, 253, 117)
        self.OBSTACE_X_CHANGE = -2
        self.obstacle_x = 500

    def collision_detection (self, obstacle_x, obstacle_height, bird_y, bottom_obstacle_height):
        if 50 <= obstacle_x <= (50 + 64):
            if bird_y <= obstacle_height or bird_y >= (bottom_obstacle_height - 64):
                return True
        return False

    def action(self, choice):
        if choice == 1:
            self.bird_y += -5
        else:
            self.bird_y += 4

        if self.bird_y <= 0:
            self.bird_y = 0
        if self.bird_y >= 571:
            self.bird_y = 571


q_table = collections.defaultdict(list)

for x in range(-371, 401):
    for y in range(-285,500):
        q_table[(x,y)] = [0 for _ in range(2)]

episode_rewards = []

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print("# : %d, epsilon : %f" %(episode, epsilon))
        print(f'{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}')

    flappy = game()
    episode_reward = 0

    for i in range(STEP):
        reward = 0
        UPPER = flappy.OBSTACLE_HEIGHT - flappy.bird_y
        LOWER = (flappy.OBSTACLE_HEIGHT + 160) - (flappy.bird_y + 64)

        obs = (UPPER, LOWER)

        if np.random.random() > epsilon:
            choice = np.argmax(q_table[obs])
        else:
            choice = np.random.randint(0,2)

        flappy.action(choice)

        flappy.obstacle_x += flappy.OBSTACE_X_CHANGE
        collision = flappy.collision_detection(flappy.obstacle_x, flappy.OBSTACLE_HEIGHT, flappy.bird_y,
                                             flappy.OBSTACLE_HEIGHT + 160)

        if collision:
            reward = -COLLISION_PENALTY
        else:
            if flappy.obstacle_x <= -10:
                flappy.obstacle_x = 500
                flappy.OBSTACLE_HEIGHT = np.random.randint(200, 400)
                reward += POINT_REWARD
            else:
                reward += ALIVE_REWARD

        UPPER = flappy.OBSTACLE_HEIGHT - flappy.bird_y
        LOWER = (flappy.OBSTACLE_HEIGHT + 160) - (flappy.bird_y + 64)

        new_obs = (UPPER, LOWER)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][choice]

        if reward == -COLLISION_PENALTY:
            new_q = -COLLISION_PENALTY
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward+DISCOUNT*max_future_q)

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