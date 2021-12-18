import numpy as np
import scipy as sp
import copy
from multiprocessing    import Process


"Defining some needed functions"
def generate_strategies_so_far(strategies_so_far):
    strategies_so_far = copy.deepcopy(strategies_so_far)
    random = 1
    for i in range(random):
        random_index           = np.random.randint(9)
        strategy               = np.zeros(9)
        strategy[random_index] = 1
        while np.sum(strategies_so_far, axis = 0)[random_index] != 0:
            random_index           = np.random.randint(9)
            strategy               = np.zeros(9)
            strategy[random_index] = 1
        strategies_so_far.append(strategy)
    return np.array(strategies_so_far)

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


"Regulating the payoff rules for player 1"
def payoff_for_player_1(matrix, matrix_shape_0, midterm_incentive):
    payoff_for_player_1 = 0
    payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 0)) :
                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j] == 0) & (matrix[i+1][j] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 0)) | ((matrix[i][j-1] == 0) & (matrix[i][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 1)) :
                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 1)) | ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 0)) :

                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j-1] == 0) & (matrix[i+1][j+1] == 1)):

                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 0)) :
                        payoff_for_player_1 += midterm_incentive
                    if ((matrix[i+1][j-1] == 0) & (matrix[i-1][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 1)) :
                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 1)) :
                        payoff_for_player_1 += midterm_incentive
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 1)):
                        payoff_for_player_1 = 1
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 1)):
                        payoff_for_player_1 = 1
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 1)) :
                        payoff_for_player_1 = 1
                    else:
                        payoff_for_player_1_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 1)) :
                        payoff_for_player_1 = 1
                    else:
                        payoff_for_player_1_smaller = 0

    return np.amax(payoff_for_player_1, payoff_for_player_1_smaller)


"Regulating the payoff rules for player 2"
def payoff_for_player_2(matrix, matrix_shape_0, midterm_incentive):
    payoff_for_player_2 = 0
    payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 0)) :
                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j] == 0) & (matrix[i+1][j] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i][j-1] == 2) & (matrix[i][j+1] == 0)) | ((matrix[i][j-1] == 0) & (matrix[i][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 2)) :
                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i][j-1] == 2) & (matrix[i][j+1] == 2)) | ((matrix[i][j-1] == 0) & (matrix[i][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 0)) :
                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j-1] == 0) & (matrix[i+1][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 0)) :
                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i+1][j-1] == 0) & (matrix[i-1][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 2)) :
                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 2)) :
                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 2)):
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i][j-1] == 2) & (matrix[i][j+1] == 2)):
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 2)):
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 2)) :
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0

    return np.amax(payoff_for_player_2, payoff_for_player_2_smaller)


" Simulating real strategies and outcomes by players"
def vectorize_strategy(matrix, strategy_int):
    vector = np.zeros(matrix.shape[0] * matrix.shape[1])
    vector[strategy_int] = 1
    return vector

def put_vectorized_strategy_of_player_1_into_matrix(matrix, vectorized_strategy):
    return matrix + vectorized_strategy.reshape((3, 3))

def put_vectorized_strategy_of_player_2_into_matrix(matrix, vectorized_strategy):
    return matrix + vectorized_strategy.reshape((3, 3)) * 2

def check_full(matrix, strategy_index):
    if matrix.reshape(9)[strategy_index] != 0:
        return "full"

def generate_from(matrix, foreseen_steps, midterm_incentive = 0, samples_selected = 1):

    X = list()
    Y = list()

    saved_matrix = matrix

    strategies   = list()

    matrix       = saved_matrix

    payoff       = 0

    for j in range(  foreseen_steps  ):

        if ((9 - foreseen_steps) + j + 1) % 2 == 1:
            if (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff == np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])).all() | (payoff == np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])).all() :
                break
            else:
                player_1_strategy = np.random.randint(matrix.shape[0] *  matrix.shape[1])
                if check_full(matrix, player_1_strategy) == "full":
                    player_1_strategy = vectorize_strategy(matrix, player_1_strategy)
                    strategies.append(np.array(player_1_strategy))
                    payoff = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1]])
                    break
                player_1_strategy = vectorize_strategy(matrix, player_1_strategy)
                matrix            = put_vectorized_strategy_of_player_1_into_matrix(matrix, player_1_strategy)
                strategies.append(np.array(player_1_strategy))
                if  (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) :
                    payoff = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0]])

        if ((9 - foreseen_steps) + j + 1) % 2 == 0:
            if (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff == np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])).all() | (payoff == np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])).all() :
                break
            else:
                player_2_strategy = np.random.randint(matrix.shape[0] * matrix.shape[1])
                if check_full(matrix, player_2_strategy) == "full":
                    player_2_strategy = vectorize_strategy(matrix, player_2_strategy)
                    strategies.append(np.array(player_2_strategy))
                    payoff = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0]])
                    break
                player_2_strategy = vectorize_strategy(matrix, player_2_strategy)
                matrix            = put_vectorized_strategy_of_player_2_into_matrix(matrix, player_2_strategy)
                strategies.append(np.array(player_2_strategy))
                if  (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) :
                    payoff = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1]])

        if (j == foreseen_steps -1) :
            if (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff == np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])).all() | (payoff == np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])).all():
                break
            else:
                payoff = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])

    X = strategies
    Y = payoff

    return np.array(X), np.array(Y)




"-----------------------------------------------------------------------------------------------"




strategies_so_far   = [np.array([0, 0, 0,
                                 0, 0, 0,
                                 0, 1, 0], dtype=float),   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                       np.array([0, 0, 0,
                                 0, 0, 1,
                                 0, 0, 0], dtype=float),   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                      ]
matrix         = turning_strategies_into_matrix(np.array(strategies_so_far))
print(matrix)
foreseen_steps = counting_remaining_steps(matrix)




for j in range(foreseen_steps):




    from Brain_for_learning import *
    network_size              = np.array([ 9 + 9 * foreseen_steps, 100, 100, 9 ])
    slope                     = 25
    alpha                     = 0.00001
    epoch_of_learning         = 150000
    drop_rate                 = 0.2
    momentum_rate             = 0.9

    Machine_01                = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)
    Machine_02                = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)
    Machine_03                = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)
    Machine_04                = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)
    Machine_05                = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)




    for i in range(epoch_of_learning):
        X, Y                                = generate_from(matrix, foreseen_steps)
        input                               = np.atleast_2d( np.zeros(9 + 9 * foreseen_steps) )
        input[:, : 9 + 9 * X.shape[0] ]     = np.atleast_2d(   np.concatenate((   matrix.flatten() * 0.5, X.flatten()   ))     )
        output                              = Y
        Machine_01.learn_batch(input, output)
        Machine_02.learn_batch(input, output)
        Machine_03.learn_batch(input, output)
        Machine_04.learn_batch(input, output)
        Machine_05.learn_batch(input, output)




    from Brain_for_deducing import *
    network_size              = np.array([ 9 + 9 * foreseen_steps, 100, 100, 9])
    value                     = -3.5
    beta                      = 0.1
    epoch_of_deducing         = 1000
    drop_rate                 = 0.75
    Machine                   = Brain(network_size, beta, epoch_of_deducing, drop_rate)

    weight_lists       = list()
    slope_lists        = list()
    weight_lists.append(Machine_01.weight_list)
    slope_lists.append (Machine_01.slope_list )
    weight_lists.append(Machine_02.weight_list)
    slope_lists.append (Machine_02.slope_list )
    weight_lists.append(Machine_03.weight_list)
    slope_lists.append (Machine_03.slope_list )
    weight_lists.append(Machine_04.weight_list)
    slope_lists.append (Machine_04.slope_list )
    weight_lists.append(Machine_05.weight_list)
    slope_lists.append (Machine_05.slope_list )




    player_movements_value = list()
    for t in range(foreseen_steps ):
        player_movements_value.append(np.array([(np.random.random(9) - 0.5) * 0.000 + value]))
    player_movements_value = np.array(player_movements_value).flatten()
    matrix_and_player_movements_value                       = np.atleast_2d( np.concatenate(  (  matrix.flatten() * 0.5   ,   player_movements_value       )  )  )

    matrix_and_player_movements_value_resistor_for_player_A = np.zeros_like(matrix_and_player_movements_value)
    for t in range(foreseen_steps ):
        if t % 2 == ((9-foreseen_steps ) % 2):
            matrix_and_player_movements_value_resistor_for_player_A[:, 9 * (t + 1) : 9 * (t + 2)] = np.atleast_2d([np.ones(9)])
    reward_for_player_A                                     = np.atleast_2d(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]))

    matrix_and_player_movements_value_resistor_for_player_B = np.zeros_like(matrix_and_player_movements_value)
    for t in range(foreseen_steps ):
        if t % 2 == np.abs((9-foreseen_steps ) % 2 - 1):
            matrix_and_player_movements_value_resistor_for_player_B[:, 9 * (t + 1) : 9 * (t + 2)] = np.atleast_2d([np.ones(9)])
    reward_for_player_B                                     = np.atleast_2d(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))




    for i in range(epoch_of_deducing ):

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




    matrix = turning_strategy_into_board(Machine.activator(player_movements_value[0, :9]), matrix)
    print(matrix)
    foreseen_steps = counting_remaining_steps(matrix)













