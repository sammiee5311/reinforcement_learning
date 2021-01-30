import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import collections
import random

#########################################
from game import Snake, Direction, Point
from model import Model, Trainer
#########################################

MAX_MEMORY = 100000
EPISODES = 1000
BATCH_SIZE = 1000
COLLISION_PENALTY = 50
POINT_REWARD = 50
SHOW_EVERY = 10
STEP = 1800

BLOCK = 10


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.9
        self.DISCOUNT = 0.95
        self.LEARNING_RATE = 0.01
        self.EPS_DECAY = 0.99998
        self.q_table = collections.deque(maxlen=MAX_MEMORY)
        self.model = Model(11, 256, 3)
        self.trainer = Trainer(self.model, self.LEARNING_RATE, self.DISCOUNT)

    def get_state(self, game):
        head = game.snake_body[0]
        point_l = Point(head.x - BLOCK, head.y)
        point_r = Point(head.x + BLOCK, head.y)
        point_u = Point(head.x, head.y - BLOCK)
        point_d = Point(head.x, head.y + BLOCK)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.fruit.x < game.snake_head.x,
            game.fruit.x > game.snake_head.x,
            game.fruit.y < game.snake_head.y,
            game.fruit.y > game.snake_head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.q_table.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.q_table) > BATCH_SIZE:
            mini_sample = random.sample(self.q_table, BATCH_SIZE)
        else:
            mini_sample = self.q_table

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        final_choice = [0, 0, 0]

        if np.random.random() > self.epsilon:
            cur_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(cur_state)
            choice = torch.argmax(prediction).item()
            final_choice[choice] = 1
        else:
            choice = np.random.randint(0, 3)
            final_choice[choice] = 1

        return final_choice


def train():
    agent = Agent()
    episode_rewards = []
    record = 0
    for episode in range(EPISODES):
        game_over = False
        episode_reward = 0
        snake = Snake()

        if episode % SHOW_EVERY == 0:
            print("# : %d, epsilon : %f, score : %d" % (episode, agent.epsilon, record))
            print(f'{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}')

        for _ in range(STEP):
            reward = 0
            old_state = agent.get_state(snake)
            final_move = agent.get_action(old_state)
            snake.action(final_move)
            collision = snake.is_collision(snake.snake_head)
            snake.snake_body.append(snake.snake_head)

            if collision:
                reward = -COLLISION_PENALTY
                game_over = True
            else:
                if snake.snake_head == snake.fruit:
                    while snake.snake_head == snake.fruit:
                        snake.fruit = Point((np.random.randint(10, snake.WIDTH) // 10) * 10,
                                           (np.random.randint(10, snake.HEIGHT) // 10) * 10)
                    snake.snake_body.append(snake.snake_head)
                    snake.score += 1
                    reward = POINT_REWARD
                else:
                    snake.snake_body.popleft()

            new_state = agent.get_state(snake)

            agent.train_short_memory(old_state, final_move, reward, new_state, game_over)

            agent.remember(old_state, final_move, reward, new_state, game_over)

            episode_reward += reward

            if game_over:
                agent.n_games += 1
                agent.train_long_memory()
                episode_rewards.append(episode_reward)
                agent.epsilon *= agent.EPS_DECAY
                if snake.score > record:
                    record = snake.score
                    agent.model.save()
                break

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg, color='indigo')
    plt.ylabel("reward on %d" % SHOW_EVERY)
    plt.xlabel("episode #")
    plt.show()


if __name__ == '__main__':
    train()