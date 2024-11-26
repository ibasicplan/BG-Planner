# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter


# Softmax function to convert raw action values to probabilities
def softmax(x):
    probs = np.exp(x - np.max(x))  # Exponentiate and normalize the values
    probs /= np.sum(probs)
    return probs


def rollout_policy_fn(board):
    """A coarse, fast version of policy_fn used in the rollout phase."""
    action_probs = np.random.rand(len(board.availables))  # Random action probabilities
    return zip(board.availables, action_probs)


# Policy function returning uniform action probabilities and a score of 0
def policy_value_fn(board):
    """A function that outputs action probabilities and score for the state."""
    action_probs = np.ones(len(board.availables)) / len(board.availables)  # Uniform probabilities
    return zip(board.availables, action_probs), 0


# Alpha-Beta search with pruning to find the optimal move
def alpha_beta_search(board, depth, alpha, beta, maximizing_player):
    """Alpha-Beta pruning search to return optimal move evaluation score."""
    if depth == 0 or board.game_end():  # Base case: evaluate at depth 0 or game end
        return board.evaluate_state()

    if maximizing_player:
        max_eval = -np.inf
        for action in board.availables:
            eval = alpha_beta_search(board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:  # Prune search tree
                break
        return max_eval
    else:
        min_eval = np.inf
        for action in board.availables:
            eval = alpha_beta_search(board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:  # Prune search tree
                break
        return min_eval


# Alpha-Beta policy to select the best move during simulations
def alpha_beta_policy(board):
    """Alpha-Beta pruning to choose move during simulation."""
    best_action = None
    best_score = -np.inf
    alpha = -np.inf
    beta = np.inf
    for action in board.availables:
        board_copy = copy.deepcopy(board)  # Make a copy of the board state
        board_copy.do_move(action)  # Apply the move
        score = alpha_beta_search(board_copy, 3, alpha, beta, False)  # Search for best score
        if score > best_score:
            best_score = score
            best_action = action  # Select the best action
    return best_action, 1.0


# TreeNode class for Alpha-Beta pruning in the Monte Carlo Tree Search (MCTS) algorithm
class TreeNodeAB(object):
    """A node in the MCTS tree, keeps track of value Q, prior P, and adjusted score u."""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # Map from actions to TreeNode
        self._n_visits = 0  # Number of visits to this node
        self._Q = 0  # Value of this node
        self._u = 0  # Adjusted score for this node
        self._P = prior_p  # Prior probability of the action

    # Expand the tree by adding children based on action-prior probabilities
    def expand(self, action_priors):
        """Expand tree by creating new children."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNodeAB(self, prob)

    # Select the next action based on exploration-exploitation trade-off
    def select(self, c_puct):
        """Select action that gives max Q + u(P)."""
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    # Update node values based on leaf evaluation
    def update(self, leaf_value):
        """Update the node value after leaf evaluation."""
        self._n_visits += 1  # Increment visit count
        self._Q += (leaf_value - self._Q) / self._n_visits  # Running average of the value

    # Recursive update for parent nodes
    def update_recursive(self, leaf_value):
        """Recursive update for all ancestor nodes."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)  # Update parent first
        self.update(leaf_value)

    # Calculate and return the value of this node based on Q and prior P
    def get_value(self, c_puct):
        """Calculate and return node value, combining Q and P adjusted by visits."""
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))  # UCB adjustment
        return self._Q + self._u

    # Check if node is a leaf (no expanded children)
    def is_leaf(self):
        """Check if node is a leaf (no expanded children)."""
        return self._children == {}

    # Check if node is the root node
    def is_root(self):
        return self._parent is None


# Monte Carlo Tree Search (MCTS) with Alpha-Beta pruning
class MCTSab(object):
    """MCTS with Alpha-Beta pruning to select actions."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Initialize MCTS with policy function and parameters."""
        self._root = TreeNodeAB(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    # Perform a single playout: select actions and propagate back the result
    def _playout(self, state):
        """Run a single playout from root to leaf, updating values."""
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next action based on UCB
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # Use Alpha-Beta pruning policy to select action
        action, _ = alpha_beta_policy(state)
        state.do_move(action)
        # Evaluate the leaf node with a rollout
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)  # Propagate evaluation back through tree

    # Rollout policy to simulate game until end
    def _evaluate_rollout(self, state, limit=1000):
        """Rollout simulation to evaluate the game state."""
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]  # Select the best action
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")
        return 1 if winner == player else -1 if winner != -1 else 0  # Return game result

    # Get most visited action from MCTS root after all playouts
    def get_move(self, state):
        """Perform all playouts and return the most visited action."""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self.pullout(state_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    # Perform a single playout and update node values
    def pullout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        action_probs, _ = self._policy(state)
        # Check for game end
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state)  # Evaluate the leaf node
        node.update_recursive(-leaf_value)

    # Update tree after each move
    def update_with_move(self, last_move):
        """Update the tree with the latest move."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNodeAB(None, 1.0)

    def __str__(self):
        return "AB MCTS"


# AI player using MCTS with Alpha-Beta pruning
class ABPlayer(object):
    """AI player using MCTS with Alpha-Beta pruning."""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTSab(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """Get the next action based on MCTS."""
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "AB MCTS {}".format(self.player)

    def name(self):
        return "AB MCTS"
