# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from utils import ModelLoader


# Function to set the learning rate for the optimizer
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Action layer of the neural network
class actionLayer(nn.Module):
    def __init__(self, board_width, board_height):
        super(actionLayer, self).__init__()
        # Convolutional layer for action probabilities
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        # Fully connected layer to calculate action scores
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)


# Proposition layer for the value estimation
class propositionLayer(nn.Module):
    def __init__(self, board_width, board_height):
        super(propositionLayer, self).__init__()
        # Convolutional layer for value estimation
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        # Fully connected layers for value prediction
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)


# The main network architecture (ASNets)
class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # Layers for action prediction
        self.actionLayer = actionLayer(board_width, board_height)
        # Layers for value prediction
        self.propositionLayer = propositionLayer(board_width, board_height)
        # Initial convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Access to action and value layers from the `actionLayer` and `propositionLayer`
        self.act_conv1 = self.actionLayer.act_conv1
        self.act_fc1 = self.actionLayer.act_fc1
        self.val_conv1 = self.propositionLayer.val_conv1
        self.val_fc1 = self.propositionLayer.val_fc1
        self.val_fc2 = self.propositionLayer.val_fc2

    def forward(self, state_input):
        # Apply the convolutional layers for feature extraction
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Action prediction layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # Value prediction layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val


# ASNets class to manage the model and training process
class ASNets():
    def __init__(self, board_width, board_height, model_params=None,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2 regularization constant

        # Initialize the neural network model
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)

        # Load model parameters if provided
        if model_params:
            self.policy_value_net.load_state_dict(model_params)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        # Load model from file if provided
        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    # Get action probabilities and value for a batch of states
    def policy_value(self, state_batch):
        if self.use_gpu:
            state_batch = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    # Policy function for a specific board state
    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    # Train the model with a batch of data
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            state_batch = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = np.array(mcts_probs)
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        # Calculate the loss (value loss + policy loss)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        # Compute entropy to measure exploration
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()

    # Get the current model parameters
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    # Save the model to a file
    def save_model(self, model_file):
        net_params = self.get_policy_param()  # Get model params
        torch.save(net_params, model_file)  # Save model to file
