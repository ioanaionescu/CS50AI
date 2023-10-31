"""
Tic Tac Toe Player
"""

import copy
import math
import random

X = "X"
O = "O"
EMPTY = None
ACCURACY = 999999

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # adding and subtracting 1 according to the element we find 
    count = sum(1 if elem == X else -1 if elem == O else 0 for row in board for elem in row)

    # considering that X starts the game, whenever the count is 1 it's O's turn
    return X if count == 0 else O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # for every EMPTY slot we add its position to the list
    return [(i, j) for i, row in enumerate(board) for j, elem in enumerate(row) if elem == EMPTY]

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    (i, j) = action
    if board[i][j] is not EMPTY:
        raise Exception('Not a valid action!')
    
    new_board = copy.deepcopy(board)

    # we add the current player's symbol to the location indicated by the action
    new_board[i][j] = player(board)
    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != EMPTY:  # Check rows
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != EMPTY:  # Check columns
            return board[0][i]
            
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]
    
    # No winner
    return None  


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # True if there is a winner or there are no more actions possible
    return winner(board) or actions(board) == [] 


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # we map the assigned value for each winner and the 0 for the default case
    # since board is terminal we can directly check the winner
    return {X : 1, O : -1}.get(winner(board), 0)


def minimax(board):
    """
    Returns the optimal action for the current player on the board. Does not take into account
    cost (as in it might choose to win in 2 moves even though a 1-move win is possible)
    """
    # [Optimisation]: Always start first corner, spare the calculations
    if board == initial_state():
        return (0, 0)

    # Using a dictionary to map players to their respective value functions
    func_map = {X: maxValue, O: minValue}
    
    # Using the dictionary to get the value function and then fetch the action
    _, action = func_map[player(board)](board, ACCURACY, float('-inf'), float('inf'))

    return action


def minValue(board, depth, alpha, beta):
    """
    Calculate the optimal value and corresponding action for the current player 
    given the board state. This function assumes the current player aims to minimize 
    the score, and will choose the best action to achieve this.
    """
    # Base cases: if at max depth or a terminal state
    if depth < 0 or terminal(board):
        return utility(board), None

    v = float('inf')
    best_action = None
    available_actions = actions(board)
    
    # Shuffle the actions for randomness
    random.shuffle(available_actions)

    for action in available_actions:
        score, _ = maxValue(result(board, action), depth - 1, alpha, beta)
        if v > score:
            v = score
            best_action = action
        # [Optimisation]: Update beta value for alpha-beta pruning
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v, best_action


def maxValue(board, depth, alpha, beta):
    """
    Calculate the optimal value and corresponding action for the current player 
    given the board state. This function assumes the current player aims to maximize 
    the score, and will choose the best action to achieve this.
    """
    # Base cases: if at max depth or a terminal state
    if depth < 0 or terminal(board):
        return utility(board), None

    v = float('-inf')
    best_action = None
    available_actions = actions(board)

    # Shuffle the actions for randomness
    random.shuffle(available_actions)

    for action in available_actions:
        score, _ = minValue(result(board, action), depth - 1, alpha, beta)
        if v < score:
            v = score
            best_action = action
        # [Optimisation]: Update alpha value for alpha-beta pruning
        alpha = max(alpha, v)
        if beta <= alpha:
            break

    return v, best_action