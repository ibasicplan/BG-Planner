# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = 400

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            if action not in state.availables:
                action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class GPAPNN_Dqn_Player(object):
    def __init__(self, policy_value_function, dqnplayer,
                 c_puct=5, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, 400)
        self._is_selfplay = is_selfplay
        self.dqnplayer = dqnplayer

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)

        if len(sensible_moves) > 0:
            if board.is_empty():
                center = (board.height // 2, board.width // 2)
                move = board.location_to_move(center)
            elif len(board.states) == 1:
                current_location = board.move_to_location(list(board.states.keys())[0])
                next_location = (current_location[0] + 1, current_location[1])
                move = board.location_to_move(next_location)
            elif len(board.states) == 2:
                locations = [board.move_to_location(move) for move in board.states.keys()]
                min_row = min(location[0] for location in locations) + 1
                max_row = max(location[0] for location in locations) - 1
                min_col = min(location[1] for location in locations) - 1
                max_col = max(location[1] for location in locations) + 1
                available_moves = [move for move in board.availables if
                                   (min_row <= board.move_to_location(move)[0] <= max_row) and (
                                               min_col <= board.move_to_location(move)[1] <= max_col)]
                available_moves = [move for move in available_moves if move not in board.states]
                if available_moves:
                    move = np.random.choice(available_moves)
                else:
                    move = np.random.choice(board.availables)
            else:
                acts, probs = self.mcts.get_move_probs(board, temp)
                move_probs[list(acts)] = probs
                if self._is_selfplay:
                    move = np.random.choice(
                        acts,
                        p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                    )
                    score1 = board.get_score()
                    opponent_move = self.predict_opponent_move(board)
                    score2 = board.get_score()
                    if score2 - score1 > 0:
                        move = np.random.choice(acts,
                                                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                        self.mcts.update_with_move(move)
                    else:
                        self.mcts.update_with_move(opponent_move)
                else:
                    move = np.random.choice(acts, p=probs)
                    self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move

        else:
            print("WARNING: the board is full")

    def predict_opponent_move(self, board):
        move = board.location_to_move(self.dqnplayer.choose_action(board, 2))
        if move not in board.availables:
            move = np.random.choice(board.availables)
        return move

    def get_action_xy(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            x = move // board.width
            y = move % board.width

            if return_prob:
                return (x, y), move_probs
            else:
                return (x, y)

    def __str__(self):
        return "GPAPNN_Dqn {}".format(self.player)

    def name(self):
        return "GPAPNN_Dqn"
