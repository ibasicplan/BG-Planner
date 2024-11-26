# -*- coding: utf-8 -*-

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import PurePlayer as MCTS_Pure
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import copy


def set_learning_rate(optimizer, lr):
    """Sets the learning rate for the optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """Defines the policy-value network architecture."""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.device = torch.device("cuda:0")

        # Common convolutional layers
        self.common_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Action policy layers
        self.policy_net = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_width * board_height, board_width * board_height),
            nn.LogSoftmax(dim=1)
        )

        # State value layers
        self.value_net = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_width * board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state_input):
        """Forward pass to calculate policy and value."""
        state_input = state_input.to(self.device)
        x = self.common_layers(state_input)
        x_act = self.policy_net(x)
        x_val = self.value_net(x)
        return x_act, x_val


class PolicyValueNet():
    """Manages the policy-value network and its training."""

    def __init__(self, board_width, board_height, model_params=None, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2 penalty coefficient
        self.device = torch.device("cuda:0")
        self.policy_value_net = self.create_net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        # Load model parameters if provided
        if model_params:
            self.policy_value_net.load_state_dict(model_params)
        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def create_net(self, board_width, board_height):
        """Creates the network based on whether GPU is used."""
        if self.use_gpu:
            return Net(board_width, board_height).to(self.device)
        else:
            return Net(board_width, board_height).to(self.device)

    def process_state_batch(self, state_batch):
        """Converts the state batch into a tensor."""
        if self.use_gpu:
            return Variable(torch.FloatTensor(np.array(state_batch)).cuda())
        else:
            return Variable(torch.FloatTensor(np.array(state_batch)))

    def policy_value(self, state_batch):
        """Calculates action probabilities and state values for a batch of states."""
        state_batch = self.process_state_batch(state_batch)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board):
        """Returns action probabilities and state value for a given board state."""
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        current_state = self.process_state_batch(current_state)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        value = value.data.cpu().numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """Performs one training step."""
        # Wrap input data into Variables
        state_batch = Variable(torch.from_numpy(np.array(state_batch)).float()).to(self.device)
        mcts_probs = Variable(torch.from_numpy(np.array(mcts_probs)).float()).to(self.device)
        winner_batch = Variable(torch.from_numpy(np.array(winner_batch)).float()).to(self.device)

        # Zero gradients and set learning rate
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # Forward pass
        log_act_probs, value = self.policy_value_net(state_batch)

        # Loss: (value_loss + policy_loss + L2 regularization)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss + self.l2_const * sum(
            torch.sum(torch.pow(p, 2)) for p in self.policy_value_net.parameters())

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        # Calculate entropy for monitoring
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        """Returns the model parameters."""
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """Saves the model parameters to a file."""
        net_params = self.get_policy_param()  # Get model parameters
        torch.save(net_params, model_file)



if __name__ == '__main__':
    # Initialize and run the training pipeline
    training_pipeline = Train()
    training_pipeline.run()
