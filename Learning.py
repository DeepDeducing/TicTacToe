import numpy as np
import scipy as sp
import copy


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

def generate_from(strategies_so_far, matrix, foreseen_steps, midterm_incentive = 0, samples_selected = 1):

    X = list()
    Y = list()

    saved_matrix = matrix

    strategies   = list()

    matrix       = saved_matrix

    payoff       = 0

    for j in range(  foreseen_steps  ):

        if (np.array(strategies_so_far).shape[0] + j + 1) % 2 == 1:
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

        if ((np.array(strategies_so_far).shape[0] + j + 1) % 2 == 0):
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


"Defining the numbers of the sets of weight matrices to be trained"
start_set     = 3
end_set       = 4
n_sets        = end_set - start_set + 1
for m in range(n_sets):


    "Importing neural network module and setting up parameters"
    from Brain_for_learning import *

    network_size              = np.array([ 9 + 9 * 4, 100, 100, 9 ])

    slope                     = 25

    alpha                     = 0.00001
    epoch_of_learning         = 1000000

    drop_rate                 = 0.2

    momentum_rate             = 0.9

    Machine                   = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)

    retrain = False
    if retrain == True:
        Machine.weight_list   = np.load("100x100_25_0.00001_1m_[" + str(start_set + m) +  "]_weight_list.npy"  , allow_pickle=True)
        Machine.slope_list    = np.load("100x100_25_0.00001_1m_[" + str(start_set + m) +  "]_slope_list.npy"   , allow_pickle=True)


    "Start learning by SGD and error backpropagation"
    for i in range(epoch_of_learning):


        print(i)


        "Setting up the present plays of the game board" "you are free to play with it"
        strategies_so_far   = [np.array([0, 0, 0,
                                         0, 0, 0,
                                         1, 0, 0], dtype=float),
                               np.array([0, 0, 0,
                                         0, 1, 0,
                                         0, 0, 0], dtype=float),
                               np.array([1, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0], dtype=float),
                              ]


        "Turning the present plays into state"
        matrix                              = turning_strategies_into_matrix(np.array(strategies_so_far))


        foreseen_steps                      = 4
        X, Y                                = generate_from(strategies_so_far, matrix, foreseen_steps)


        "State and one-hotted actions"
        input                               = np.atleast_2d( np.zeros(9 + 9 * 4) )
        index                               = X.shape[0]
        input[:, : 9 + 9 * index ]          = np.atleast_2d(   np.concatenate((   matrix.flatten() * 0.5, X.flatten()   ))     )


        "Quantifying reward"
        output = Y


        "Learning for a iteration/epoch"
        Machine.learn_batch( input , output)


    "Saving a set of trained matrices for deducing phase"
    np.save("100x100_25_0.00001_1m_[" + str(start_set + m) +  "]_weight_list"             , Machine.weight_list        )
    np.save("100x100_25_0.00001_1m_[" + str(start_set + m) +  "]_slope_list"              , Machine.slope_list         )



