# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """A coarse, fast version of the policy function used in the rollout phase."""
    # Perform random rollouts for action selection
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """A function that takes in a board state and outputs a list of (action, probability)
    tuples and a score for the state."""
    # Return uniform action probabilities and a score of 0 (for MCTS)
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNodePure(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # A map from action to TreeNode
        self._n_visits = 0  # Number of visits to this node
        self._Q = 0  # Node value (Q)
        self._u = 0  # Exploration bonus (u)
        self._P = prior_p  # Prior probability

    def expand(self, action_priors):
        """Expand the tree by creating new children for each possible action.
        action_priors: List of (action, prior_probability) tuples from the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNodePure(self, prob)

    def select(self, c_puct):
        """Select the child node with the maximum value (Q + u), where u is the exploration bonus.
        c_puct: Exploration parameter controlling the impact of the prior.
        Returns: A tuple (action, next_node).
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update the node values based on the leaf evaluation.
        leaf_value: The value of the subtree from the current player's perspective.
        """
        # Increment visit count
        self._n_visits += 1
        # Update Q using a running average
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Update the node and its ancestors recursively."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)  # Update parent first
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        Combines the leaf value Q and the exploration bonus u, based on the prior probability P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))  # UCB formula
        return self._Q + self._u

    def is_leaf(self):
        """Check if the node is a leaf (i.e., has no children)."""
        return self._children == {}

    def is_root(self):
        """Check if the node is the root of the tree."""
        return self._parent is None


class MCTSpure(object):
    """A simple implementation of Monte Carlo Tree Search (MCTS) using pure Monte Carlo simulations."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Initialize MCTS with a policy function, exploration parameter c_puct, and number of playouts.
        policy_value_fn: Function that returns (action, probability) tuples and a state evaluation.
        c_puct: Exploration parameter controlling exploration vs exploitation.
        n_playout: Number of playouts (simulations) per move.
        """
        self._root = TreeNodePure(None, 1.0)  # Initialize the root node
        self._policy = policy_value_fn  # Policy function
        self._c_puct = c_puct  # Exploration parameter
        self._n_playout = n_playout  # Number of simulations to run

    def _playout(self, state):
        """Run a single playout (simulation) from the root node to a leaf, then propagate the result back.
        State is modified in-place, so a copy of the state must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():  # If it's a leaf, stop expanding
                break
            # Select the next action greedily based on exploration/exploitation
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Apply the policy function to get action probabilities
        action_probs, _ = self._policy(state)
        # Check if the game has ended
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)  # Expand the tree if the game isn't over
        # Evaluate the leaf node using a rollout
        leaf_value = self._evaluate_rollout(state)
        # Update the node and its ancestors based on the evaluation
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Evaluate the game state by performing a random rollout until the game ends.
        Returns +1 if the current player wins, -1 if the opponent wins, or 0 for a tie.
        """
        player = state.get_current_player()  # Get the current player
        for i in range(limit):
            end, winner = state.game_end()  # Check if the game has ended
            if end:
                break
            # Randomly select an action
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break (game limit reached), issue a warning
            print("WARNING: rollout reached move limit")

        # Return result based on the winner
        if winner == -1:  # Tie
            return 0
        else:
            return 1 if winner == player else -1  # Win or loss

    def get_move(self, state):
        """Run all playouts (simulations) and return the most visited action.
        state: The current game state
        Return: The selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)  # Copy state to avoid modification during simulation
            self._playout(state_copy)
        # Return the action with the most visits
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree based on the last move, preserving known subtree information."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNodePure(None, 1.0)

    def __str__(self):
        return "Pure MCTS"


class PurePlayer(object):
    """AI player based on MCTS."""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTSpure(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """Get the next action based on the current board state."""
        sensible_moves = board.availables  # Available moves
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Pure MCTS {}".format(self.player)

    def name(self):
        return "Pure MCTS"
