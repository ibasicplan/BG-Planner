import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm

from game import Board, Game
from mcts_pure import PurePlayer
from GPAPNN_pure import GPAPNNPlayer
from baseline import PolicyValueNet
from model import ASNets
from mcts_alphaZero import AlphazeroPlayer
import csv
import random
from torch.autograd import Variable
from collections import deque

# Set device for computation (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n = 5
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)
width, height = 15, 15

# Convert action index to coordinates
def farse(move, board_width):
    x = move // board_width
    y = move % board_width
    return (x, y)

# Define the DQN model architecture
class DQN(nn.Module):
    def __init__(self, width, height):
        super(DQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * width * height, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, width * height)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store a transition in memory
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample a batch of transitions from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Agent that interacts with the environment and learns
class DQNAgent():
    def __init__(self, input_channels, width, height, memory_capacity=100000, batch_size=64, gamma=0.99,
                 learning_rate=1e-3, target_update_freq=1000, threshold_initial=None, threshold_decay=50000, min_threshold=1):
        self.device = device
        self.width = width
        self.height = height
        self.input_channels = input_channels
        self.action_space = [(i, j) for i in range(width) for j in range(height)]
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        # Initialize the main and target networks
        self.model = DQN(width, height).to(self.device)
        self.target_model = DQN(width, height).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target network to evaluation mode

        # Define optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.1)

        # Initialize epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 50000  # Total steps for epsilon decay
        self.step_done = 0

        # Initialize threshold parameters
        if threshold_initial is None:
            self.threshold = width / 2
        else:
            self.threshold = threshold_initial
        self.threshold_decay = threshold_decay
        self.min_threshold = min_threshold

    def choose_action(self, state, epsilon=None):
        """Choose an action based on epsilon-greedy strategy"""
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            # Choose a random action
            action = random.choice(self.action_space)
        else:
            # Choose the best action based on Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, 4, width, height)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_index = q_values.argmax().item()
            action = farse(action_index, self.width)
        return action

    def update_epsilon(self):
        """Decay epsilon for exploration-exploitation trade-off"""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1.0 - self.epsilon_min) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_threshold(self):
        """Decay threshold for better convergence"""
        if self.threshold > self.min_threshold:
            self.threshold -= (self.threshold - self.min_threshold) / self.threshold_decay
            self.threshold = max(self.threshold, self.min_threshold)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Learn from experiences stored in memory"""
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert data to tensors
        states = torch.stack([torch.from_numpy(s).float() for s in states]).to(self.device)  # (batch, 4, width, height)
        actions = torch.tensor([a[0] * self.width + a[1] for a in actions]).long().to(self.device)  # (batch,)
        rewards = torch.tensor(rewards).float().to(self.device).unsqueeze(1)  # (batch, 1)
        next_states = torch.stack([torch.from_numpy(ns).float() for ns in next_states]).to(self.device)  # (batch, 4, width, height)
        dones = torch.tensor(dones).float().to(self.device).unsqueeze(1)  # (batch, 1)

        # Calculate Q-values for current states
        q_values = self.model(states).gather(1, actions.unsqueeze(1))  # (batch, 1)

        # Calculate expected Q-values using target network
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)  # (batch, 1)
            expected_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)  # (batch, 1)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Update epsilon for exploration-exploitation trade-off
        self.update_epsilon()

        return loss.item()

    def replace_target_network(self):
        """Copy parameters from the main network to the target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        """Save the model parameters to the specified path"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model parameters from the specified path"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Function to count consecutive pieces in a given direction
def count_consecutive_pieces(board, move, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_count = 0
    for dx, dy in directions:
        count = 1
        for dir in [-1, 1]:
            x, y = move[0] + dir * dx, move[1] + dir * dy
            while 0 <= x < board.width and 0 <= y < board.height and board.states.get((x, y)) == player:
                count += 1
                x += dir * dx
                y += dir * dy
        max_count = max(count, max_count)
    return max_count

# Ensure the directory for saving models exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Play the game and update the agent with rewards and actions
def play_game(agent, opponent, num_games=100):
    for game_num in range(num_games):
        board = Board(width, height)
        game = Game(board)
        state = board.get_state()
        done = False
        while not done:
            action = agent.choose_action(state)
            reward, next_state, done = game.step(action, agent)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

            # Learn after each action
            agent.learn()

        if game_num % agent.target_update_freq == 0:
            agent.replace_target_network()

    print(f"Completed {num_games} games.")
