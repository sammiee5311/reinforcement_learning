import numpy as np
import time
import torch
import torch.nn as nn
import random

#########################################
from game import Snake, Direction, Point
from model import Model
#########################################

BLOCK = 20

def get_state(game):
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


def get_action(model, state):
    final_choice = [0, 0, 0]
    cur_state = torch.tensor(state, dtype=torch.float)
    prediction = model(cur_state)
    choice = torch.argmax(prediction).item()
    final_choice[choice] = 1

    return final_choice


snake = Snake()
model = Model(11, 256, 3)
is_running = True
state_dict = torch.load('./model/model.pth')
model.load_state_dict(state_dict)

while is_running:
    state = get_state(snake)

    choice = get_action(model, state)
    snake.action(choice)

    collision = snake.is_collision(snake.snake_head)

    snake.snake_body.insert(0, snake.snake_head)
    snake.snake_body.pop()

    if collision:
        is_running = False

    if snake.snake_head == snake.fruit:
        while snake.snake_head == snake.fruit:
            snake.fruit = Point((random.randint(0, snake.WIDTH - BLOCK) // BLOCK) * BLOCK,
                                (random.randint(0, snake.HEIGHT - BLOCK) // BLOCK) * BLOCK)
        snake.snake_body.insert(0, snake.snake_head)
        snake.score += 1

    time.sleep(0.08)
    snake.display()
