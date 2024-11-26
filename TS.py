# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter

# Rollout policy function to generate random action probabilities
def rollout_policy_fn(board):
    action_probs = np.random.rand(len(board.availables))  # Random probabilities for available actions
    return zip(board.availables, action_probs)

# Policy value function to calculate action probabilities and evaluate board state
def policy_value_fn(board):
    action_probs = np.ones(len(board.availables)) / len(board.availables)  # Equal probabilities for all actions
    score = 0
    found_special_pattern = False

    # Check for special board patterns (e.g., five in a row, live four, etc.)
    for i in range(board.width):
        for j in range(board.height):
            if is_double_three(board, i, j):  # Check for double three pattern
                score = 1
                found_special_pattern = True
            elif is_live_four(board, i, j):  # Check for live four pattern
                score = 1
                found_special_pattern = True
            elif is_five_in_a_row(board, i, j):  # Check for five in a row
                score = 1
                found_special_pattern = True

    # Return action probabilities and score if a special pattern is found, else return None and score 0
    if not found_special_pattern:
        return None, 0
    return zip(board.availables, action_probs), score


# Check if there are five consecutive stones in a row
def is_five_in_a_row(board, i, j):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Possible directions (horizontal, vertical, diagonal)
    for dx, dy in directions:
        count = 0
        for step in range(-4, 5):  # Check range of steps from -4 to +4
            x, y = i + step * dx, j + step * dy
            if 0 <= x < board.width and 0 <= y < board.height:
                if board.states.get((x, y)) == board.current_player:
                    count += 1
                else:
                    count = 0
            if count == 5:
                return True
    return False

# Check if there is a live four (four stones in a row with open ends)
def is_live_four(board, i, j):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 0
        for step in range(-4, 5):  # Check range of steps from -4 to +4
            x, y = i + step * dx, j + step * dy
            if 0 <= x < board.width and 0 <= y < board.height:
                if board.states.get((x, y)) == board.current_player:
                    count += 1
                else:
                    count = 0
            if count == 4:
                return True
    return False

# Check if there is a sequence of three consecutive stones in a row
def is_three(board, i, j):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 0
        for step in range(-3, 4):  # Check range of steps from -3 to +3
            x, y = i + step * dx, j + step * dy
            if 0 <= x < board.width and 0 <= y < board.height:
                if board.states.get((x, y)) == board.current_player:
                    count += 1
                else:
                    count = 0
            if count == 3:
                return True
    return False

# Check if there are two separate sequences of three consecutive stones (double three)
def is_double_three(board, i, j):
    count = 0
    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for step in range(-3, 4):  # Check range of steps from -3 to +3
            x, y = i + step * dx, j + step * dy
            if 0 <= x < board.width and 0 <= y < board.height:
                if is_three(board, x, y):  # Check if the position has a three-in-a-row pattern
                    count += 1
            if count == 2:
                return True
    return False

class TreeNodePure(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # A dictionary to store child nodes
        self._n_visits = 0  # Visit count
        self._Q = 0  # Value estimate
        self._u = 0  # Exploration bonus
        self._P = prior_p  # Prior probability for this node

    def expand(self, action_priors):
        """Expand the tree by creating new child nodes for each action-prior pair."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNodePure(self, prob)

    def select(self, c_puct):
        """Select the child node that maximizes the action value Q plus bonus u."""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update the node's value from the evaluation of its leaf node."""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits  # Running average of leaf values

    def update_recursive(self, leaf_value):
        """Recursively update this node and its ancestors with the leaf value."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)  # Update parent node with the opposite value
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value of this node (Q + exploration bonus u)."""
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if this node is a leaf node (no children)."""
        return self._children == {}

    def is_root(self):
        """Check if this node is the root of the tree."""
        return self._parent is None


class MCTSpure(object):
    """A simple implementation of Monte Carlo Tree Search (MCTS)."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNodePure(None, 1.0)  # Root node of the MCTS tree
        self._policy = policy_value_fn  # The policy function
        self._c_puct = c_puct  # Exploration constant
        self._n_playout = n_playout  # Number of playouts per move

    def _playout(self, state):
        """Perform a playout starting from the root node and update the tree."""
        node = self._root
        while(1):
            if node.is_leaf():  # If leaf node, stop selecting
                break
            action, node = node.select(self._c_puct)  # Select action based on UCT
            state.do_move(action)

        action_probs, _ = self._policy(state)
        end, winner = state.game_end()  # Check if game has ended
        if not end:
            # Expand tree if no special pattern found
            if action_probs is None:
                return
            node.expand(action_probs)
        # Evaluate leaf node through rollout
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)  # Backpropagate value to root

    def _evaluate_rollout(self, state, limit=1000):
        """Evaluate a leaf node by performing a random rollout."""
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")  # Warning if the limit is reached
        if winner == -1:  # Tie
            return 0
        else:
            return 1 if winner == player else -1  # Return 1 for win, -1 for loss

    def get_move(self, state):
        """Get the best move based on MCTS."""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)  # Simulate the state
            self._playout(state_copy)
        if self._root._children:
            return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]
        else:
            return state.get_legal_move()  # Default move if no simulations performed

    def update_with_move(self, last_move):
        """Update the MCTS tree with the last move."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNodePure(None, 1.0)

    def __str__(self):
        return "Reverse"


class Reverse(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTSpure(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """Get the next action for the AI player."""
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            if move is None:
                print("WARNING: no special pattern found")
                return None
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Reverse {}".format(self.player)
