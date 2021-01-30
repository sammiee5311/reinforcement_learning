import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save(self, file_name='model.pth'):
        PATH = './model'

        if not os.path.exists(PATH):
            os.makedirs(PATH)

        file_PATH = os.path.join(PATH, file_name)
        torch.save(self.state_dict(), file_PATH)


class Trainer:
    def __init__(self, model, LEARNING_RATE, DISCOUNT):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(game_over)):
            new_obs = reward[idx]
            if not game_over[idx]:
                new_obs = reward[idx] + self.DISCOUNT * torch.max(self.model(next_state))

            target[idx][torch.argmax(action).item()] = new_obs

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()

        self.optimizer.step()