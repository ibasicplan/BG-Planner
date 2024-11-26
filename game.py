# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
import copy


class Board(object):
    """Board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))  # Board width
        self.height = int(kwargs.get('height', 15))  # Board height
        self.states = {}  # Stores the board state (key: position, value: player)
        self.n_in_row = int(kwargs.get('n_in_row', 5))  # Number of pieces in a row to win
        self.players = [1, 2]  # Player identifiers (1 and 2)

    def init_board(self, start_player=0):
        """Initialize the board state for a new game"""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('Board size must be at least n_in_row')
        self.current_player = self.players[start_player]  # Start with the specified player
        self.availables = list(range(self.width * self.height))  # List of available moves
        self.states = {}  # Reset the board state
        self.last_move = -1  # No move made initially

    def move_to_location(self, move):
        """
        Convert a move index to a (row, column) location on the board
        e.g., move 5 maps to location (1, 2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """Convert a (row, column) location to a move index"""
        if len(location) != 2:
            return -1
        h, w = location
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """Return the board state from the perspective of the current player.
        The state has shape (4, width, height) where:
        - square_state[0] represents current player's pieces
        - square_state[1] represents opponent's pieces
        - square_state[2] represents the last move made
        - square_state[3] indicates the current player to move
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # Current player to play
        return square_state[:, ::-1, :]  # Flip the state to align with current player

    def state_vector(self):
        """Return the board state as a vector of shape (2 * width * height)"""
        vector_state = np.zeros(2 * self.width * self.height)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            vector_state[moves[players == self.players[0]]] = 1.0
            vector_state[moves[players == self.players[1]] + self.width * self.height] = 1.0
        return vector_state

    def do_move(self, move):
        """Update the board state by making a move"""
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self):
        """Check if there is a winner on the board"""
        width, height, states, n = self.width, self.height, self.states, self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))  # List of moves made
        if len(moved) < self.n_in_row * 2 - 1:  # Not enough pieces placed to determine a winner
            return False, -1

        # Check horizontal, vertical, and diagonal lines for a winner
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # Check horizontal
            if w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1:
                return True, player

            # Check vertical
            if h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1:
                return True, player

            # Check main diagonal
            if w in range(width - n + 1) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1:
                return True, player

            # Check anti-diagonal
            if w in range(n - 1, width) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1:
                return True, player

        return False, -1

    def game_end(self):
        """Check if the game has ended (either by a win or a tie)"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):  # If no more moves available, it's a tie
            return True, -1
        return False, -1

    def get_current_player(self):
        """Return the current player"""
        return self.current_player

    def is_empty(self):
        """Check if the board is empty (no moves made yet)"""
        return len(self.states) == 0

    def evaluate_state(self):
        """Evaluate the board state and return a score based on the current player's position"""
        score = 0
        lines = []

        # Convert board state to 2D array for easier evaluation
        board = [[0] * self.width for _ in range(self.height)]
        for move, player in self.states.items():
            h, w = self.move_to_location(move)
            board[h][w] = player

        # Add all rows
        lines.extend(board)

        # Add all columns
        lines.extend(zip(*board))

        # Add diagonals
        lines.append([board[i][i] for i in range(len(board))])
        lines.append([board[i][len(board) - i - 1] for i in range(len(board))])

        # Score based on 5-in-a-row patterns
        for line in lines:
            for i in range(len(line) - 4):
                segment = line[i:i + 5]
                if all(x == self.current_player for x in segment):
                    score += 1000  # Five consecutive pieces of current player
                elif all(x == self.current_player for x in segment[:-1]) and segment[-1] == 0:
                    score += 100  # Four consecutive pieces + one empty space
                elif all(x == self.current_player for x in segment[:-2]) and all(x == 0 for x in segment[-2:]):
                    score += 10  # Three consecutive pieces + two empty spaces
                elif all(x == self.current_player for x in segment[:-3]) and all(x == 0 for x in segment[-3:]):
                    score += 1  # Two consecutive pieces + three empty spaces

        return score

    def get_legal_move(self):
        """Return a random available move"""
        return random.choice(self.availables)


class Game(object):
    """Game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and display game information"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')  # Print column headers
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')  # Print row index
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')  # Print player 1's piece
                elif p == player2:
                    print('O'.center(8), end='')  # Print player 2's piece
                else:
                    print('_'.center(8), end='')  # Empty space
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """Start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) or 1 (player2 first)')
        start_player = random.choice([0, 1])  # Randomly choose who starts
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            if current_player == 1:
                move = player_in_turn.get_action(self.board)
                self.board.do_move(move)
                end, winner = self.board.game_end()
                if end:
                    if is_shown:
                        if winner != -1:
                            print("Game end. Winner is", players[winner])
                        else:
                            print("Game end. Tie")
                    return winner
            if current_player == 2:
                move = player_in_turn.get_action(self.board)
                self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """Self-play between the agent and itself"""
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())  # Store state
            mcts_probs.append(move_probs)  # Store move probabilities
            current_players.append(self.board.current_player)  # Store current player
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0  # Mark winner's perspective
                    winners_z[np.array(current_players) != winner] = -1.0  # Mark loser
                player.reset_player()  # Reset player for next game
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
