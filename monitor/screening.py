from glob import glob1
import numpy as np
from attackgraph.simulation import series_sim
from attackgraph import file_op as fp
import os
import re

"""
This file conducts simulations for combined game. The strategies produced by different heuristics are stored in the 
directory combined_game. Function screen loads the names of strategies from those files. Combined with run.py, we can
simulate the payoff matrix of the combined game.
"""


def_str_abs_path = '/attackgraph/defender_strategies/'
att_str_abs_path = '/attackgraph/attacker_strategies/'

def preprocess_file(file):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(file)
    parts[1::2] = map(int, parts[1::2])
    return parts


def screen(paths):
    '''
    screen function monitors if new strategy has been added to the strategy sets of different methods.
    :param paths: a dict recording paths to the main directory of different methods. {"baselines": path, "rew", path}
    :return:
    '''
    str_dict_def = {}
    str_dict_att = {}
    for method in paths:
        print("Checking strategies in ", method)
        path_def = paths[method] + def_str_abs_path
        path_att = paths[method] + att_str_abs_path

        if not fp.isExist(path_def):
            raise ValueError("Defender's strategy path does not exist.")
        elif not fp.isExist(path_att):
            raise ValueError("Attacker's strategy path does not exist.")

        # Remove the uniform strategy
        str_dict_def[method] = sorted(glob1(path_def, "*.pkl"), key=preprocess_file)[1:]
        str_dict_att[method] = sorted(glob1(path_att, "*.pkl"), key=preprocess_file)[1:]
        if len(str_dict_def[method]) == 0 or len(str_dict_att[method]) == 0:
            raise ValueError("There is no strategy in the strategy directory.")

    return str_dict_def, str_dict_att


def check_new_str(methods, new_dict_def, new_dict_att, old_dict_def, old_dict_att):
    """
    Check if there is new strategy produced by different method.
    :param methods: different meta-strategy solver.
    :param new_dict_def: defender's current str dict.
    :param new_dict_att: attacker's current str dict.
    :param old_dict_def: defender's old str dict.
    :param old_dict_att: attacker's old str dict.
    :return:
    """
    new_att = {}
    new_def = {}
    for method in methods:
        new_def[method] = new_dict_def[method] - old_dict_def[method]
        new_att[method] = new_dict_att[method] - old_dict_att[method]
        if len(new_def[method]) == 0 and len(new_att[method]) == 0:
            print(method + " has not produced a new strategy.")
        else:
            print(method + ":defender's new str is ", new_def[method])
            print(method + ":attacker's new str is ", new_att[method])

    return new_def, new_att

# TODO: check all the index.
def simulate_payoff(co_payoff_matrix_def,
                    co_payoff_matrix_att,
                    str_book_def,
                    str_book_att,
                    new_def,
                    new_att,
                    save_path='./payoff_data/'):
    """
    This function simulates the partial payoff matrix of the combined game given newly produced strategies.
    :param co_payoff_matrix: The partial payoff matrix of the combined game
    :param str_book: a book recording the method, its corresponding strategy name and position.
                    e.g. {position: str_name}
    :param new_def: newly produced defender's strategies.
    :param new_att: newly produced attacker's strategies.
    :return:
    """

    path = os.getcwd() + '/data/game.pkl'
    game = fp.load_pkl(path)

    env = game.env
    num_episodes = game.num_episodes

    # set positions.
    new_att = list(new_att)
    new_def = list(new_def)

    num_str_def, num_str_att = np.shape(co_payoff_matrix_def)
    num_new_str_def = len(new_def)
    num_new_str_att = len(new_att)
    new_dim_def = num_str_def + num_new_str_def
    new_dim_att = num_str_att + num_new_str_att

    for i in np.arange(num_new_str_def):
        idx = i + num_str_def
        if idx in str_book_def.keys():
            raise ValueError("idx already exists.")
        str_book_att[idx] = new_def[i]

    for i in np.arange(num_new_str_att):
        idx = i + num_str_att
        if idx in str_book_att.keys():
            raise ValueError("idx already exists.")
        str_book_att[idx] = new_att[i]

    position_row = []
    position_col = []
    # add column first.
    for i in np.arange(num_new_str_att):
        position_col_list = []
        for k in np.arange(num_str_def):
            position_col_list.append((k,num_str_att+i))

        position_row.append(position_col_list)

    # Then add row.
    for i in np.arange(num_new_str_def):
        position_row_list = []
        for k in np.arange(new_dim_att):
            position_row_list.append((num_new_str_def+i,k))

        position_row.append(position_row_list)


    att_col = []
    att_row = []
    def_col = []
    def_row = []

    for list in position_col:
        subcol_def = []
        subcol_att = []
        for pos in list:
            idx_def, idx_att = pos
            # TODO: reset the load path
            aReward, dReward = series_sim(env, game, str_book_att[idx_att], str_book_def[idx_def], num_episodes)
            subcol_att.append(aReward)
            subcol_def.append(dReward)

        att_col.append(subcol_att)
        def_col.append(subcol_def)

    for list in position_row:
        subrow_def = []
        subrow_att = []
        for pos in list:
            idx_def, idx_att = pos
            # TODO: reset the load path
            aReward, dReward = series_sim(env, game, str_book_att[idx_att], str_book_def[idx_def], num_episodes)
            subrow_att.append(aReward)
            subrow_def.append(dReward)

        att_row.append(subrow_att)
        def_row.append(subrow_def)

    for col in att_col:
        col = np.reshape(np.round(np.array(col),2),newshape=(num_str_def,1))
        co_payoff_matrix_att = add_col(co_payoff_matrix_att, col)

    for row in att_row:
        rol = np.round(np.array(row),2)[None]
        co_payoff_matrix_att = add_row(co_payoff_matrix_att, rol)

    for col in def_col:
        col = np.reshape(np.round(np.array(col), 2), newshape=(num_str_def, 1))
        co_payoff_matrix_def = add_col(co_payoff_matrix_def, col)

    for row in def_row:
        rol = np.round(np.array(row), 2)[None]
        co_payoff_matrix_def = add_row(co_payoff_matrix_def, rol)

    return co_payoff_matrix_def, co_payoff_matrix_att, str_book_def, str_book_att


def add_col(matrix, col):
    matrix = np.append(matrix, col, 1)
    return matrix

def add_row(matrix, row):
    matrix = np.append(matrix, row, 0)
    return matrix































