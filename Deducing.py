import numpy as np
import scipy as sp
import copy


"Defining some needed functions"
def turning_strategies_into_matrix(strategies_so_far):
    matrix = np.zeros((3, 3))
    matrix = matrix.flatten()
    for i in  range(strategies_so_far.shape[0]):
        if i % 2 == 0:
            matrix[np.argmax(strategies_so_far[i])] = 1
        if i % 2 == 1:
            matrix[np.argmax(strategies_so_far[i])] = 2
    matrix = matrix.reshape((3, 3))
    return matrix

def counting_remaining_steps(matrix):
    matrix = matrix.flatten()
    number_of_total_zeros = 0
    for i in range(matrix.shape[0]):
        if matrix[i] == 0:
            number_of_total_zeros += 1
    return number_of_total_zeros


"Setting up the present plays of the game board" "you are free to play with it"
strategies_so_far =         [np.array([0, 0, 0,
                                       0, 0, 0,
                                       1, 0, 0], dtype=float),
                             np.array([0, 0, 0,
                                       0, 1, 0,
                                       0, 0, 0], dtype=float),
                             np.array([1, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0], dtype=float),
                             ]
strategies_so_far = np.array(strategies_so_far)

matrix = turning_strategies_into_matrix(strategies_so_far)
print("The current state of the board is:")
print(matrix)

remaining_steps      = counting_remaining_steps(matrix)
print("The remaining steps are:")
print(remaining_steps)

foreseen_steps = 4
print("The foreseen steps are:")
print(foreseen_steps)




"Setting up given states, initialized actions and optinal reward"
from Brain_for_deducing import *

network_size              = np.array([ 9 + 9 * 4, 100, 100, 9])

value                     = -3.5

beta                      = 0.01
epoch_of_deducing         = 10000

drop_rate                 = 0.2

Machine                   = Brain(network_size, beta, epoch_of_deducing, drop_rate)


"Importing sets of trained weight matrices from the learning phase"
weight_lists       = list()
slope_lists        = list()
n_sets = 1
for i in range(n_sets):
    weight_list        = np.load("100x100_25_0.00001_1m_[" + str(2 + i + 1) +  "]_weight_list.npy"  , allow_pickle=True)   #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    slope_list         = np.load("100x100_25_0.00001_1m_[" + str(2 + i + 1) +  "]_slope_list.npy"    , allow_pickle=True)   #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)


"Setting up given states, initialized actions and optinal reward for player A and B"
player_movements_value = list()
for j in range(foreseen_steps):
    player_movements_value.append(np.array([(np.random.random(9) - 0.5) * 0.000 + value]))
player_movements_value = np.array(player_movements_value).flatten()
matrix_and_player_movements_value                       = np.atleast_2d( np.concatenate(  (  matrix.flatten() * 0.5   ,   player_movements_value       )  )  )

matrix_and_player_movements_value_resistor_for_player_A = np.zeros_like(matrix_and_player_movements_value)
for t in range(foreseen_steps):
    if t % 2 == (strategies_so_far.shape[0] % 2):
        matrix_and_player_movements_value_resistor_for_player_A[:, 9 * (t + 1) : 9 * (t + 2)] = np.atleast_2d([np.ones(9)])
reward_for_player_A                                     = np.atleast_2d(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]))

matrix_and_player_movements_value_resistor_for_player_B = np.zeros_like(matrix_and_player_movements_value)
for t in range(foreseen_steps):
    if t % 2 == np.abs(strategies_so_far.shape[0] % 2 - 1):
        matrix_and_player_movements_value_resistor_for_player_B[:, 9 * (t + 1) : 9 * (t + 2)] = np.atleast_2d([np.ones(9)])
reward_for_player_B                                     = np.atleast_2d(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))


"Start dedcuing by MWM-SGD and error backpropagation by minimax"
for k in range(Machine.epoch_of_deducing ):

    random_index     = np.random.randint(np.asarray(weight_lists).shape[0])
    weight_list      = weight_lists[random_index]
    slope_list       = slope_lists[random_index]

    matrix_and_player_movements_value                     = Machine.deduce_batch(matrix_and_player_movements_value ,
                                                                                 matrix_and_player_movements_value_resistor_for_player_A,
                                                                                 reward_for_player_A,
                                                                                 weight_list,
                                                                                 slope_list
                                                                                 )

    matrix_and_player_movements_value                     = Machine.deduce_batch(matrix_and_player_movements_value ,
                                                                                 matrix_and_player_movements_value_resistor_for_player_B,
                                                                                 reward_for_player_B,
                                                                                 weight_list,
                                                                                 slope_list
                                                                                 )


"Deciding real/final action based on optimzed initial actions"
player_movements_value = matrix_and_player_movements_value[:, 9:]


def turning_strategy_into_board(strategy, board):
    ones = np.zeros_like(board)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] != 0:
                ones[i][j] = 1
    strategy  = strategy.reshape((3, 3)) - strategy.reshape((3, 3)) * ones
    point     = np.argmax(strategy.flatten())
    new_board = board.flatten()
    if np.sum(ones) % 2 == 0:
        new_board[point] = 1
    else:
        new_board[point] = 2
    return (new_board.reshape((3, 3)))


print(np.round(Machine.activator(player_movements_value[0, :9]), 2).reshape((3, 3))  )
print("The final movement for Deep Deducing is:")
print(turning_strategy_into_board(Machine.activator(player_movements_value[0, :9]), matrix) )




