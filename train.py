from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from GPAPNN_pure import GPAPNNPlayer
from mcts_pure import PurePlayer as MCTS_Pure
from mcts_AB import ABPlayer as MCTS_AB
from mcts_Minimax import MinimaxPlayer as MCTS_Minimax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import matplotlib.pyplot as plt


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class actionLayer(nn.Module):
    def __init__(self, in_channels, board_width, board_height):
        super(actionLayer, self).__init__()
        self.act_conv1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)


class propositionLayer(nn.Module):
    def __init__(self, in_channels, board_width, board_height):
        super(propositionLayer, self).__init__()
        self.val_conv1 = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)


class MultiLayerNet(nn.Module):
    def __init__(self, board_width, board_height):
        super(MultiLayerNet, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.actionLayers = nn.ModuleList([
            actionLayer(128, board_width, board_height) for _ in range(3)
        ])

        self.propositionLayers = nn.ModuleList([
            propositionLayer(128, board_width, board_height) for _ in range(3)
        ])

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        act_outputs = []
        val_outputs = []

        for i in range(3):
            x_act = F.relu(self.actionLayers[i].act_conv1(x))
            x_act_flat = x_act.view(-1, 4 * self.board_width * self.board_height)
            x_act_out = F.log_softmax(self.actionLayers[i].act_fc1(x_act_flat), dim=1)
            act_outputs.append(x_act_out)

            x_val = F.relu(self.propositionLayers[i].val_conv1(x))
            x_val_flat = x_val.view(-1, 2 * self.board_width * self.board_height)
            x_val_hidden = F.relu(self.propositionLayers[i].val_fc1(x_val_flat))
            x_val_out = F.tanh(self.propositionLayers[i].val_fc2(x_val_hidden))
            val_outputs.append(x_val_out)

        return act_outputs[-1], val_outputs[-1]


class ASNets():
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4
        if self.use_gpu:
            self.policy_value_net = MultiLayerNet(board_width, board_height).cuda()
        else:
            self.policy_value_net = MultiLayerNet(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            print(f"loading model from {model_file}")
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

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

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            state_batch = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


class TrainPipeline():
    def __init__(self, init_model=None, board_width=15,
                 board_height=15, n_in_row=5, learn_rate=2e-3, lr_multiplier=1.0,
                 temp=1.0, n_playout=400, c_puct=5, buffer_size=10000,
                 batch_size=512, play_batch_size=1, epochs=5, kl_targ=0.02,
                 check_freq=300, game_batch_num=1500, best_win_ratio=0.0,
                 pure_mcts_playout_num=1000,
                 increase_per_iteration=100
                 ):
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = learn_rate
        self.lr_multiplier = lr_multiplier  # adaptively adjust the learning rate based on KL
        self.temp = temp  # the temperature param
        self.n_playout = n_playout  # num of simulations for each move
        self.c_puct = c_puct
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = play_batch_size
        self.epochs = epochs  # num of train_steps for each update
        self.kl_targ = kl_targ
        self.check_freq = check_freq
        self.game_batch_num = game_batch_num
        self.best_win_ratio = best_win_ratio
        self.pure_mcts_playout_num = pure_mcts_playout_num
        self.increase_per_iteration = increase_per_iteration
        if init_model:
            self.asnets = ASNets(self.board_width,
                                 self.board_height,
                                 model_file=init_model)
        else:
            self.asnets = ASNets(self.board_width,
                                 self.board_height)
        self.mcts_player = GPAPNNPlayer(self.asnets.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def rotate_and_flip(self, state, mcts_prob):
        for i in range(1, 5):
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
            yield equi_state, np.flipud(equi_mcts_prob).flatten()
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            yield equi_state, np.flipud(equi_mcts_prob).flatten()

    def get_equi_data(self, play_data):
        extend_data = [(equi_state, equi_mcts_prob, winner)
                       for state, mcts_prob, winner in play_data
                       for equi_state, equi_mcts_prob in self.rotate_and_flip(state, mcts_prob)]
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.asnets.policy_value(state_batch)

        for i in range(self.epochs):
            if self.asnets.use_gpu:
                state_batch_tensor = Variable(torch.FloatTensor(state_batch).cuda())
                mcts_probs_tensor = Variable(torch.FloatTensor(mcts_probs).cuda())
                winner_batch_tensor = Variable(torch.FloatTensor(winner_batch).cuda())
            else:
                state_batch_tensor = Variable(torch.FloatTensor(state_batch))
                mcts_probs_tensor = Variable(torch.FloatTensor(mcts_probs))
                winner_batch_tensor = Variable(torch.FloatTensor(winner_batch))
            self.asnets.optimizer.zero_grad()
            for param_group in self.asnets.optimizer.param_groups:
                param_group['lr'] = self.learn_rate * self.lr_multiplier

            log_act_probs, value = self.asnets.policy_value_net(state_batch_tensor)
            value_loss = F.mse_loss(value.view(-1), winner_batch_tensor)
            policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_act_probs, 1))
            loss = value_loss + policy_loss
            loss.backward()
            self.asnets.optimizer.step()
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )
            loss_value = loss.item()
            entropy_value = entropy.item()
            new_probs, new_v = self.asnets.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        with open('results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([kl, self.lr_multiplier, loss_value, entropy_value, explained_var_old, explained_var_new])
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss_value,
                        entropy_value,
                        explained_var_old,
                        explained_var_new))
        self.pure_mcts_playout_num += self.increase_per_iteration
        return loss_value, entropy_value

    def policy_evaluate(self, n_games=3):
        current_mcts_player = GPAPNNPlayer(self.asnets.policy_value_fn,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                 n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)

        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                      pure_mcts_player,
                                      start_player=i % 2,
                                      is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        for i in range(self.game_batch_num):
            self._run_game_batch(i)

    def _run_game_batch(self, i):
        self.collect_selfplay_data(self.play_batch_size)
        print(f"batch i:{i + 1}, episode_len:{self.episode_len}")
        self._update_policy_if_needed()
        self._evaluate_and_save_model_if_needed(i)

    def _update_policy_if_needed(self):
        if len(self.data_buffer) > self.batch_size:
            loss, entropy = self.policy_update()

    def _evaluate_and_save_model_if_needed(self, i):
        if (i + 1) % self.check_freq == 0:
            self._evaluate_and_save_model(i)

    def _evaluate_and_save_model(self, i):
        print(f"current self-play batch: {i + 1}")
        win_ratio = self.policy_evaluate()
        self.asnets.save_model('./current_policy15.model')
        self._update_best_policy_if_needed(win_ratio)

    def _update_best_policy_if_needed(self, win_ratio):
        if win_ratio > self.best_win_ratio:
            print("New best policy!!!!!!!!")
            self.best_win_ratio = win_ratio
            self.asnets.save_model('./best_policy15.model')
            self._increase_playout_num_if_needed()

    def _increase_playout_num_if_needed(self):
        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 8000:
            self.pure_mcts_playout_num += 1000
            self.best_win_ratio = 0.0


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
