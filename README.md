# BG-Planner

In **BG-Planner**, GraphNet starts with three convolutional layers to extract abstract features from the board state, with dimensions of:

- **32×15×15**
- **64×15×15**
- **128×15×15**

This is followed by alternating **action** and **proposition** layers, with each action layer having dimensions of **4×15×15** and each proposition layer **2×15×15**.

In our experiments, the network includes three action and three proposition layers, with flexibility to adjust the layer count for different scenarios.


# 1 GraphNet 

## 1.1 Input Layer
The input data shape is `(batch_size, 4, 15, 15)`, comprising 4 channels:
1. **Channel 1**: Current player's pieces.
2. **Channel 2**: Opponent's pieces.
3. **Channel 3**: Last move's position.
4. **Channel 4**: Indicates whose turn it is (1 for first player, -1 for second).

## 1.2 Convolution Layers
- **Conv1**: `nn.Conv2d(4, 32, kernel_size=3, padding=1)`, extracts basic features with 32 output channels.
- **Conv2**: `nn.Conv2d(32, 64, kernel_size=3, padding=1)`, deepens feature extraction with 64 channels.
- **Conv3**: `nn.Conv2d(64, 128, kernel_size=3, padding=1)`, further refines features with 128 channels.

## 1.3 Action Layer
This layer uses `Conv2d` followed by `Linear` layers to output action probabilities. It has a total of 3,375 neurons.

## 1.4 Proposition Layer
Processes features related to value evaluation. Outputs a score between -1 and 1, with a total of 1,545 neurons.

## 1.5 Forward Pass
- **Shared Convolution**: Three convolution layers to extract board features.
- **Action Layer**: Computes action probabilities.
- **Proposition Layer**: Outputs game value evaluation.

## 1.6 Training Details
- **Optimizer**: Adam with a learning rate of \(2 \times 10^{-3}\) and weight decay \(1 \times 10^{-4}\).
- **Data Augmentation**: Includes rotations and flips to enhance generalization.
- **Self-Play**: Uses Monte Carlo Tree Search (MCTS) for generating training data.

## 1.7 Hyperparameters
The hidden layer sizes for action and proposition layers are fixed at 225 and 64, respectively.

## 1.8 Total Neurons
- **Total neurons**: 55,320 neurons across all layers.

# 2 DQN 

The DQN (Deep Q-Network) consists of a neural network, replay memory, and an agent that interacts with the environment. The neural network is composed of convolutional layers for feature extraction and fully connected layers to calculate Q-values for each action. It uses an epsilon-greedy exploration strategy to balance exploration and exploitation, and a replay memory to store transitions for training.

## 2.1 Key Components

- **Neural Network**: Includes convolutional and fully connected layers, with ReLU activations and a final output layer for Q-values.
- **Replay Memory**: Stores transitions (state, action, reward, next_state) and allows for random sampling during training.
- **DQN Agent**: Encapsulates the model, optimizer, target model, and exploration strategy. It also manages the training process with methods like `choose_action`, `learn`, and `store_transition`.
- **Training Process**: The agent interacts with the environment, stores transitions, samples batches for training, and updates the model using a target network to stabilize learning.

# 3 Knowledge-based Search Tactic

## Overview
To enhance the performance of BG-Planner, we introduce the knowledge-based Tabu Search (TS), which leverages external commonsense knowledge. This tactic enables the agent to utilize knowledge of "Winning Sequences," a scenario where victory is guaranteed within a limited number of moves.

## Approach
The TS iterates through possible actions, guided by the knowledge of winning strategies. A proposed evaluation function ranks these sequences to prioritize actions most likely to lead to a win, optimizing the agent’s decision-making and improving its strategic performance.


# 4 AlphaZero Model 

## 4.1 Input Layer
The input layer consists of four channels representing various board states:
- Channel 1: Current player's piece positions.
- Channel 2: Opponent's piece positions.
- Channel 3: Last move's position.
- Channel 4: Player's turn indicator.

## 4.2 Convolutional Layers
The model uses three convolutional layers to extract features from the board:
- Conv1: Extracts basic local features (32 filters, 3x3 kernel).
- Conv2: Extracts deeper features (64 filters, 3x3 kernel).
- Conv3: Extracts high-level features for strategy and value evaluation (128 filters, 3x3 kernel).

## 4.3 Policy & Value Networks
- **Policy Network**: Outputs a probability distribution of potential actions.
- **Value Network**: Provides a scalar value indicating the board's winning probability.

## 4.4 Training Details
The model is trained using:
- **Loss Functions**: Value loss (MSE) and policy loss (Cross-Entropy).
- **Optimizer**: Adam with L2 regularization.
- **Data Augmentation**: Rotations and flips to increase sample diversity.
- **Self-play**: Training data is generated from self-play with MCTS.

## 4.5 Evaluation & Termination
- **Model Evaluation**: Performance is tested by playing against MCTS.
- **Termination Criteria**: Training ends when the model reaches optimal performance with a win rate of 100% and a predefined number of MCTS simulations.

