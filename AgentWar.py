# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle
from game import Board, Game
from mcts_alphaZero import AlphazeroPlayer
from GPAPNN_pure import GPAPNNPlayer
from mcts_AB import ABPlayer
from mcts_Minimax import MinimaxPlayer
from mcts_pure import PurePlayer
from TS import Reverse
from utils import ModelLoader
import torch
from model import ASNets
from baseline import PolicyValueNet


class Human(object):
    """
    Human player class for interacting with the game.
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        """
        Prompts the human player for their move and validates it.
        """
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(mode, az, mcts, reverse):
    """
    Main function to run different game modes.
    """
    n = 5  # Number of pieces in a row to win
    width, height = 15, 15  # Board dimensions
    model_file = './model/GPAPNN/0604_gpapnn_train_best_policy.model'
    alphazero_model = './model/Alphazero/0603_alphazero.h5'

    # Load models
    policy_param = torch.load(model_file)
    alphazero_param = torch.load(alphazero_model)

    # Initialize AI policy network
    policy_net = ASNets(width, height, policy_param)
    Alphazero_param = PolicyValueNet(width, height, alphazero_param)
    player = GPAPNNPlayer(policy_net.policy_value_fn)

    # Choose MCTS player based on input
    if az == 'true':
        pass
        mcts_player = AlphazeroPlayer(Alphazero_param.policy_value_fn, c_puct=5, n_playout=4000)
    else:
        if mcts == 'pure':
            mcts_player = PurePlayer(c_puct=5, n_playout=5000)
        elif mcts == 'ab':
            mcts_player = ABPlayer(c_puct=5, n_playout=18000)
        elif mcts == 'minimax':
            mcts_player = MinimaxPlayer(c_puct=5, n_playout=400)

    # Game modes
    if mode == 'ai2ai':
        try:
            board = Board(width=width, height=height, n_in_row=n)
            game = Game(board)
            game.start_play(player, mcts_player, start_player=0, is_shown=1)
        except KeyboardInterrupt:
            print('\n\rquit')

    elif mode == 'human2gpapnn':
        try:
            board = Board(width=width, height=height, n_in_row=n)
            game = Game(board)
            human = Human()  # Human player instance
            game.start_play(player, human, start_player=0, is_shown=1)  # AI goes first
            # game.start_play(player, human, start_player=1, is_shown=1)  # AI goes second
        except KeyboardInterrupt:
            print('\n\rquit')


import argparse

if __name__ == '__main__':
    # Argument parser for different configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ai2ai', choices=['ai2ai', 'human2gpapnn'],
                        help='Game mode. Default is ai2ai.')
    parser.add_argument('--az', default='false', choices=['true', 'false'],
                        help='alphazero mode. Default is pure.')
    parser.add_argument('--mcts', default='pure', choices=['pure', 'ab', 'minimax'],
                        help='MCTS mode. Default is pure.')
    parser.add_argument('--reverse', default='true', choices=['true', 'false'],
                        help='Reverse mode. Default is false.')
    args = parser.parse_args()

    # Run the game multiple times
    for _ in range(5):
        run(args.mode, args.az, args.mcts, args.reverse)
