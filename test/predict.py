# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pickle
from mcts_alphaZero import AlphazeroPlayer
from GPAPNN_pure import GPAPNNPlayer
from mcts_AB import ABPlayer
from mcts_Minimax import MinimaxPlayer
from mcts_pure import PurePlayer
from utils import ModelLoader
from mcts_alphaZero_new import DQNPlayer, DqnNet
import random
from model import ASNets
from baseline import PolicyValueNet
import numpy as np
import csv

n = 5
width, height = 15, 15
Model_file = 'best_policy.model'
Alphazero_model = 'alphazero.h5'
Dqnmodel= 'model_ep5500.h5'
episodes = 50000
max_steps = width * height
save_model_frequency = 500
target_update = 100
threshold = 4
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.input_size = 2 * width * height
        self.output_size = width * height
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            raise ValueError(f"Not enough elements in memory to sample {batch_size} elements")
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent():
    def __init__(self, input_size, output_size, memory_capacity, batch_size, gamma, cols, rows):
        self.device = torch.device("cuda:0")
        self.input_size = input_size
        self.output_size = output_size
        self.memory = ReplayMemory(memory_capacity)
        self.action_space = [(i, j) for i in range(width) for j in range(height)]
        self.batch_size = batch_size
        self.gamma = gamma
        self.cols = cols
        self.rows = rows
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 50000
        self.threshold = width / 2
        self.threshold_decay = 50000
        self.min_threshold = 1

    def reset_action_space(self):
        self.action_space = [(i, j) for i in range(width) for j in range(height)]

    def choose_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            state_array = state.state_vector()
            state_vector = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
            q_values = self.model(state_vector)
            action_index = np.argmax(q_values.cpu().detach().numpy())
            if action_index >= len(self.action_space):
                action = random.choice(self.action_space)
            else:
                action = self.action_space[action_index]
        else:
            action = random.choice(self.action_space)
        return action

    def update_epsilon(self):
        self.epsilon -= (self.epsilon - self.epsilon_end) / self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)

    def update_threshold(self):
        self.threshold -= (self.threshold - self.min_threshold) / self.threshold_decay
        self.threshold = max(self.threshold, self.min_threshold)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.from_numpy(s).float() for s in states]).to(self.device)
        actions = torch.stack([torch.tensor(a).long() for a in actions]).unsqueeze(-1).to(self.device)
        rewards = torch.stack([torch.tensor(r).float() for r in rewards]).to(self.device)
        next_states = torch.stack([torch.from_numpy(ns).float() for ns in next_states]).to(self.device)
        dones = torch.stack([torch.tensor(d).float() for d in dones]).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def train_with_mcts_data(self, mcts_data):
        for state, action, reward, next_state, done in mcts_data:
            self.store_transition(state, action, reward, next_state, done)
            self.learn()

    def replace_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

dqnAgent = DQNAgent(width*height, width*height, 10000, 32, 0.99, width, height)
dqnAgent.load_model(Dqnmodel)

dqn_predictor = dqnAgent

class Board(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.states = {}
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def state_vector(self):
        vector_state = np.zeros(2*self.width*self.height)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            vector_state[move_curr] = 1.0
            vector_state[move_oppo] = -1.0
        return vector_state

def evaluate_state(board):
    state = board.current_state()
    return np.argmax(dqn_predictor.model(torch.FloatTensor(state).unsqueeze(0).to(dqn_predictor.device)).cpu().detach().numpy())

def play_game():
    board = Board(width=width, height=height)
    board.init_board(start_player=0)

    done = False
    while not done:
        action = dqn_predictor.choose_action(board.current_state(), dqn_predictor.epsilon)
        print(action)
        location = board.move_to_location(action)
        if location in board.availables:
            move = board.location_to_move(location)
            board.states[move] = board.current_player
            board.availables.remove(move)
            board.last_move = move
            done = True
        else:
            print("Invalid move!")
        return done
