import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import collections

style.use('ggplot')

EPISODES = 15000
MOVE_PENALTY = 1
ANSWER_REWARD = 50
HALF_REWARD = 20
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 500
MIN_VALUES = -300

LEARNING_RATE = 0.001
DISCOUNT = 0.95
STEP = 600


class number_game:
    def __init__(self):

        self.first_digit = np.random.randint(0,10)
        self.second_digit = np.random.randint(0,10)

    def __sub__(self, other):
        return self.first_digit - other.first_digit, self.second_digit - other.second_digit

    def action(self, choice):
        if choice == 0:
            pass
        elif choice == 1:
            self.plus_one(0)
        elif choice == 2:
            self.plus_one(1)
        elif choice == 3:
            self.minus_one(0)
        else:
            self.minus_one(1)

    def minus_one(self, choice):
        if choice == 0:
            self.first_digit -= 1
        elif choice == 1:
            self.second_digit -= 1

        if self.first_digit < 0:
            self.first_digit = 0
        if self.second_digit < 0:
            self.second_digit = 0

    def plus_one(self, choice):
        if choice == 0:
            self.first_digit += 1
        elif choice == 1:
            self.second_digit += 1

        if self.first_digit > 9:
            self.first_digit = 9
        if self.second_digit > 9:
            self.second_digit = 9


q_table = collections.defaultdict(list)

for d1 in range(-9, 10):
    for d2 in range(-9, 10):
        q_table[(d1,d2)] = [0 for _ in range(5)]


episode_rewards = []

for episode in range(EPISODES):
    player = number_game()
    answer = number_game()

    if episode % SHOW_EVERY == 0:
        print("# : %d, epsilon : %f" %(episode, epsilon))
        print(f'{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}')

    if episode % 1000 == 0 and episode != 0:
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(STEP):
        obs = player - answer
        if np.random.random() > epsilon:
            choice = np.argmax(q_table[obs])
        else:
            choice = np.random.randint(0,5)

        player.action(choice)
        if player.first_digit == answer.first_digit and player.second_digit == answer.second_digit:
            reward = ANSWER_REWARD
        else:
            reward = MOVE_PENALTY

        new_obs = player - answer
        max_future_q = np.argmax(q_table[new_obs])
        current_q = q_table[obs][choice]

        if reward == ANSWER_REWARD:
            new_q = ANSWER_REWARD
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward+DISCOUNT*max_future_q)

        q_table[obs][choice] = new_q

        if show:
            print(player.first_digit,answer.first_digit,player.second_digit,answer.second_digit)

        episode_reward += reward
        if reward == ANSWER_REWARD or reward <= MIN_VALUES:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

print(q_table)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel("reward %d" % SHOW_EVERY)
plt.xlabel("episode #")
plt.show()
