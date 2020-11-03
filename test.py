import numpy as np
import pickle
import pygame


class game:
    def __init__(self):
        pygame.init()
        self.SCREEN = pygame.display.set_mode((500, 750))  # Setting the display
        self.startFont = pygame.font.Font('freesansbold.ttf', 32)

        self.BACKGROUND_IMAGE = pygame.image.load('background.jpg')

        self.BIRD_IMAGE = pygame.image.load('bird1.png')
        self.bird_x = 50
        self.bird_y = 300
        self.bird_y_change = 0

        self.OBSTACLE_WIDTH = 70
        self.OBSTACLE_HEIGHT = np.random.randint(200, 400)
        self.OBSTACLE_COLOR = (211, 253, 117)
        self.OBSTACE_X_CHANGE = -2
        self.obstacle_x = 500

    def display_bird(self,x, y):
        self.SCREEN.blit(self.BIRD_IMAGE, (x, y))

    def display_obstacle(self,height):
        pygame.draw.rect(self.SCREEN, self.OBSTACLE_COLOR, (self.obstacle_x, 0, self.OBSTACLE_WIDTH, height))
        bottom_obstacle_height = 635 - height - 160
        pygame.draw.rect(self.SCREEN, self.OBSTACLE_COLOR, (self.obstacle_x, 635, self.OBSTACLE_WIDTH, - bottom_obstacle_height))

    def collision_detection (self, obstacle_x, obstacle_height, bird_y, bottom_obstacle_height):
        if 50 <= obstacle_x <= (50 + 64):
            if bird_y <= obstacle_height or bird_y >= (bottom_obstacle_height - 64):
                return True
        return False

    def action(self, choice):
        if choice == 1:
            self.bird_y_change = -5
        else:
            self.bird_y_change = 4


with open('q_table4.pickle', "rb") as file:
    q_table = pickle.load(file)


while True:
    flappy = game()
    while True:
        flappy.SCREEN.fill((0, 0, 0))
        flappy.SCREEN.blit(flappy.BACKGROUND_IMAGE, (0, 0))

        UPPER = flappy.OBSTACLE_HEIGHT - flappy.bird_y
        LOWER = (flappy.OBSTACLE_HEIGHT + 160) - (flappy.bird_y + 64)

        obs = (UPPER, LOWER)

        flappy.display_obstacle(flappy.OBSTACLE_HEIGHT)
        flappy.display_bird(flappy.bird_x, flappy.bird_y)

        pygame.display.update()

        choice = np.argmax(q_table[obs])

        flappy.action(choice)

        flappy.bird_y += flappy.bird_y_change
        if flappy.bird_y <= 0:
            flappy.bird_y = 0
        if flappy.bird_y >= 571:
            flappy.bird_y = 571

        flappy.obstacle_x += flappy.OBSTACE_X_CHANGE
        collision = flappy.collision_detection(flappy.obstacle_x, flappy.OBSTACLE_HEIGHT, flappy.bird_y,
                                               flappy.OBSTACLE_HEIGHT + 160)

        if collision:
            break
        if flappy.obstacle_x <= -10:
            flappy.obstacle_x = 500
            flappy.OBSTACLE_HEIGHT = np.random.randint(200, 400)