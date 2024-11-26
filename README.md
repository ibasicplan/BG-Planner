## BG-Planner

In **BG-Planner**, GraphNet starts with three convolutional layers to extract abstract features from the board state, with dimensions of:

- **32×15×15**
- **64×15×15**
- **128×15×15**

This is followed by alternating **action** and **proposition** layers, with each action layer having dimensions of **4×15×15** and each proposition layer **2×15×15**.

In our experiments, the network includes three action and three proposition layers, with flexibility to adjust the layer count for different scenarios.


# GraphNet Overview

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

