# MIT 6.034 Lab 3: Games
# Written by Dylan Holmes (dxh), Jessica Noss (jmn), and 6.034 staff

from game_api import *
from boards import *
INF = float('inf')

def is_game_over_connectfour(board) :
    "Returns True if game is over, otherwise False."
    if all(board.is_column_full(col) for col in range(board.num_cols)):
        return True
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True
    return False

def next_boards_connectfour(board) :
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []
    next_moves = []
    for col in range(board.num_cols):
        if not board.is_column_full(col):
            b = board.add_piece(col)
            next_moves.append(b)
    return next_moves

def endgame_score_connectfour(board, is_current_player_maximizer) :
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            if is_current_player_maximizer:
                return -1000
            else:
                return 1000
    if all(board.is_column_full(col) for col in range(board.num_cols)):
        return 0

def endgame_score_connectfour_faster(board, is_current_player_maximizer) :
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    if all(board.is_column_full(col) for col in range(board.num_cols)):
        return 0
    winner_pieces = board.count_pieces(False)
    score = lambda x: float(-300/49)*x**2 + 100*x+1600
    all_pieces = board.count_pieces()
    if all_pieces == board.num_cols * board.num_rows:
        if is_current_player_maximizer:
            return -1000
        else:
            return 1000
    else:
        if is_current_player_maximizer:
            if winner_pieces == 4:
                return -2000
            else:
                return -int(score(winner_pieces))
        else:
            if winner_pieces == 4:
                return 2000
            else:
                return int(score(winner_pieces))

def heuristic_connectfour(board, is_current_player_maximizer) :
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    current_chains = board.get_all_chains(True)
    prev_chains = board.get_all_chains(False)
    current_score = score(current_chains)
    prev_score = score(prev_chains)
    if is_current_player_maximizer:
        return sorted((-999, -(prev_score - current_score), 999))[1]
    else:
        return sorted((-999, (prev_score - current_score), 999))[1]

def score(chains):
    keeper = {1:0, 2:0, 3:0, 4:0}
    for c in chains:
        keeper[len(c)] += 1
    return sum(k**2*(keeper[k]+10) for k in keeper)

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

state_mine = AbstractGameState(snapshot = PLAYER_ONE1_WON,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### PART 2 ###########################################
# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    # res = move_sequence(GAME1, [2,3])

    (max_path, max_score, eval_static) = dfs(state, [state], 0, [], 0)
    return (max_path[:len(max_path)-1], max_score, eval_static)


def dfs(start_state, path, max_score, max_path, eval_static):
    if start_state.is_game_over():
        score = abs(start_state.get_endgame_score())
        eval_static+=1
        if score > max_score:
            max_score = score
            max_path = path+[start_state]
        if not start_state.generate_next_states():
            return (max_path, max_score, eval_static)
    for node in start_state.generate_next_states():
        (max_path, max_score, eval_static) = dfs(node, path+[node], max_score, max_path, eval_static)
    return (max_path, max_score, eval_static)


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    (max_score, max_path, eval_static) = minimax(state, [], maximize, 0)
    return (max_path[::-1], max_score, eval_static)

def minimax(start_state, path, maximizer, eval_static):
    if start_state.is_game_over():
        score = start_state.get_endgame_score(maximizer)
        eval_static+=1
        return score, [start_state], eval_static
    if maximizer:
        alpha = -INF
        for node in start_state.generate_next_states():
            score, p, eval_static = minimax(node, path, not(maximizer), eval_static)
            if score > alpha:
                alpha = score
                path = p + [start_state]
        return alpha, path, eval_static
    else:
        beta = INF
        for node in start_state.generate_next_states():
            score, p, eval_static = minimax(node, path, not(maximizer), eval_static)
            if score < beta:
                beta = score
                path = p + [start_state]
        return beta, path, eval_static

def print_path(p, des):
    print des
    for i in p:
        print i.get_snapshot()
        print '-----------------------------------------'

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    "Performs standard minimax search.  Same return type as dfs_maximizing."
    (max_score, max_path, eval_static) = minimax_standard(state, [], maximize, 0, heuristic_fn, 0, depth_limit)
    #GAME = AbstractGameState(BOARD_UHOH, is_game_over_connectfour, next_boards_connectfour, endgame_score_connectfour)
    #print_path(move_sequence(GAME, [4,5]), 'TEST 28')
    #print_path(max_path[::-1], 'MY ANS')
    return (max_path[::-1], max_score, eval_static)

def minimax_standard(start_state, path, maximizer, eval_static, h_fn, depth, depth_limit):
    if start_state.is_game_over():
        score = start_state.get_endgame_score(maximizer)
        eval_static+=1
        return score, [start_state], eval_static
    else:
        if depth == depth_limit:
            score = h_fn(start_state.get_snapshot(), maximizer)
            eval_static+=1
            return score, [start_state], eval_static
    if maximizer:
        alpha = -INF
        for node in start_state.generate_next_states():
            score, p, eval_static = minimax_standard(node, path, not(maximizer), eval_static, h_fn, depth+1, depth_limit)
            if score > alpha:
                alpha = score
                path = p + [start_state]
        return alpha, path, eval_static
    else:
        beta = INF
        for node in start_state.generate_next_states():
            score, p, eval_static = minimax_standard(node, path, not(maximizer), eval_static, h_fn, depth+1, depth_limit)
            if score < beta:
                beta = score
                path = p + [start_state]
        return beta, path, eval_static

# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1.  Try increasing the value of depth_limit to see what happens:

pretty_print_dfs_type(minimax_search(state_NEARLY_OVER, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    "Performs minimax with alpha-beta pruning.  Same return type as dfs_maximizing."
    (max_score, max_path, eval_static) = alphabeta(state, [], maximize, alpha, beta, 0, heuristic_fn, 0, depth_limit)
    return (max_path[::-1], max_score, eval_static)

def alphabeta(start_state, path, maximizer, alpha, beta, eval_static, h_fn, depth, depth_limit):
    if start_state.is_game_over():
        score = start_state.get_endgame_score(maximizer)
        eval_static+=1
        return score, [start_state], eval_static
    else:
        if depth == depth_limit:
            score = h_fn(start_state.get_snapshot(), maximizer)
            eval_static+=1
            return score, [start_state], eval_static

    if maximizer:
        for node in start_state.generate_next_states():
            score, p, eval_static = alphabeta(node, path, not(maximizer), alpha, beta, eval_static, h_fn, depth+1, depth_limit)
            if score > alpha:
                alpha = score
                path = p + [start_state]
            if alpha >= beta:
                break
        return alpha, path, eval_static
    else:
        for node in start_state.generate_next_states():
            score, p, eval_static = alphabeta(node, path, not(maximizer), alpha, beta, eval_static, h_fn, depth+1, depth_limit)
            if score < beta:
                beta = score
                path = p + [start_state]
            if alpha >= beta:
                break
        return beta, path, eval_static

# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4.  Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

pretty_print_dfs_type(minimax_search_alphabeta(state_NEARLY_OVER, heuristic_fn=heuristic_connectfour, depth_limit=1))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()   # TA Note: Use this to store values.
    depth = 1
    while depth <= depth_limit:
        anytime_value.set_value(minimax_search_alphabeta(state = state, depth_limit = depth, heuristic_fn = heuristic_fn, maximize = maximize))
        depth+=1
    #anytime_value.pretty_print()
    return anytime_value

# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4.  Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

#print progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4)


##### PART 3: Multiple Choice ##################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Chunchun Wu"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "7"
WHAT_I_FOUND_INTERESTING = "Helper functions are useful"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = "None"
