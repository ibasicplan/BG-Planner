# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter


# Softmax function to convert raw action values to probabilities
def softmax(x):
    probs = np.exp(x - np.max(x))  # Exponentiate and normalize values
    probs /= np.sum(probs)
    return probs


# A simple rollout policy function used in the rollout phase (random actions)
def rollout_policy_fn(board):
    """A coarse, fast version of policy_fn used in the rollout phase."""
    action_probs = np.random.rand(len(board.availables))  # Random action probabilities
    return zip(board.availables, action_probs)


# Policy function returning uniform action probabilities and a score of 0
def policy_value_fn(board):
    """A function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state."""
    action_probs = np.ones(len(board.availables)) / len(board.availables)  # Uniform probabilities
    return zip(board.availables, action_probs), 0


# Minimax algorithm for search and evaluation
def minimax(board, depth, maximizing_player):
    """Use the Minimax algorithm to search and return the evaluation score of the best move."""
    if depth == 0 or board.game_end():  # Base case: evaluate at depth 0 or game end
        return board.evaluate_state()

    if maximizing_player:
        max_eval = -np.inf
        for action in board.availables:
            board_copy = copy.deepcopy(board)  # Copy the board state
            board_copy.do_move(action)  # Apply the move
            eval = minimax(board_copy, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = np.inf
        for action in board.availables:
            board_copy = copy.deepcopy(board)
            board_copy.do_move(action)
            eval = minimax(board_copy, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval


# Minimax policy for selecting moves during simulation
def minimax_policy(board):
    """Use Minimax to select the best move during simulation."""
    best_action = None
    best_score = -np.inf
    for action in board.availables:
        board_copy = copy.deepcopy(board)  # Copy the board state
        board_copy.do_move(action)  # Apply the move
        score = minimax(board_copy, 3, False)  # Search for best score
        if score > best_score:
            best_score = score
            best_action = action  # Select the best action
    return best_action, 1.0


# TreeNodeMinimax class for the MCTS tree with Minimax search
class TreeNodeMinimax(object):
    """A node in the MCTS tree. Each node keeps track of its value Q,
    prior probability P, and its visit-count-adjusted prior score u."""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # Map of action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._c_puct = 1.0  # Exploration parameter

    # Expand tree by creating children based on action-prior probabilities
    def expand(self, action_priors):
        """Expand tree by creating new children."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNodeMinimax(self, prob)

    # Select the next action based on exploration-exploitation trade-off
    def select(self, c_puct):
        """Select action among children that gives maximum Q + bonus u(P)."""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    # Update node values after evaluating a leaf
    def update(self, leaf_value):
        """Update node values from leaf evaluation."""
        self._n_visits += 1  # Increment visit count
        self._Q += (leaf_value - self._Q) / self._n_visits  # Running average of value

    # Recursive update for all ancestor nodes
    def update_recursive(self, leaf_value):
        """Recursive update for all ancestor nodes."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)  # Update parent first
        self.update(leaf_value)

    # Calculate and return the value of this node (Q + u)
    def get_value(self, c_puct):
        """Calculate and return the value of this node."""
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))  # UCB adjustment
        return self._Q + self._u

    # Check if node is a leaf (no children expanded)
    def is_leaf(self):
        """Check if node is a leaf."""
        return self._children == {}

    # Check if node is the root node
    def is_root(self):
        return self._parent is None


# MCTSMinimax class implementing Monte Carlo Tree Search with Minimax
class MCTSminimax(object):
    """A simple implementation of MCTS with Minimax."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Initialize MCTS with policy function and parameters."""
        self._root = TreeNodeMinimax(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    # Perform a single playout: select actions and propagate results back through tree
    def _playout(self, state):
        """Run a single playout from root to leaf, and propagate the result back."""
        node = self._root
        while True:
            if node.is_leaf():  # Stop if leaf node
                break
            # Greedily select next action based on UCB
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action, _ = minimax_policy(state)
        state.do_move(action)

        # Evaluate leaf node and update tree
        end, winner = state.game_end()
        if not end:
            action, _ = minimax_policy(state)
            action_probs = [(action, 1.0)]  # Action probabilities
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state)  # Rollout evaluation
        node.update_recursive(-leaf_value)  # Update node values

    # Evaluate the game state using the rollout policy
    def _evaluate_rollout(self, state, limit=1000):
        """Evaluate the state using a random rollout."""
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")

        if winner == -1:  # Tie
            return 0
        else:
            return 1 if winner == player else -1  # Win or lose

    # Get the most visited move after all playouts
    def get_move(self, state):
        """Perform all playouts and return the most visited action."""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self.pullout(state_copy)

        if self._root._children:
            return max(self._root._children.items(),
                       key=lambda act_node: act_node[1]._n_visits)[0]
        else:
            return state.get_legal_move()  # Default move if no action is selected

    # Perform a playout and update tree values
    def pullout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Select next action based on UCB
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    # Update tree with the latest move
    def update_with_move(self, last_move):
        """Step forward in the tree with the last move."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNodeMinimax(None, 1.0)

    def __str__(self):
        return "Minimax MCTS"


# AI player using MCTS with Minimax
class MinimaxPlayer(object):
    """AI player based on MCTS with Minimax."""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTSminimax(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Minimax MCTS {}".format(self.player)

    def name(self):
        return "Minimax MCTS"
