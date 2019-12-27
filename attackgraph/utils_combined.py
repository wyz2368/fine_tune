from glob import glob1
import numpy as np
# from attackgraph.simulation import series_sim
from attackgraph import file_op as fp
import os
import re
import joblib
from baselines.deepq.load_action import load_action_class


# ALL paths.
# (1) Paths to the strategies of different heuristics.
def_str_abs_path = '/defender_strategies/'
att_str_abs_path = '/attacker_strategies/'


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

        print("Defender's load path is ", path_def)
        print("Attacker's load path is ", path_att)

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


def scope_finder(policy_path):
    if not fp.isExist(policy_path):
        raise ValueError("Policy does not exist.")
    loaded_params = joblib.load(os.path.expanduser(policy_path))
    scope = iter(loaded_params).__next__().split('/')[0]
    return scope

# Load all policies into a dictionary.
def load_policies(game, child_partition, identity):
    if identity == 0: # load defender's policies.
        name = def_str_abs_path + 'def_str_epoch'
    elif identity == 1:
        name = att_str_abs_path + 'att_str_epoch'
    else:
        raise ValueError("identity is not correct")

    str_dict = {}

    path = os.getcwd() + '/combined_game/'

    for key in child_partition:
        for i in np.arange(1, child_partition[key]+1):
            # nn = "RS/attacker_strategies/def_str_epoch2.pkl"
            nn = key + name + str(i+1) + '.pkl'

            uniform_flag = False
            if "epoch1.pkl" in nn:
                uniform_flag = True

            load_path = path + nn

            # Strategies are kept as a tuple with parameters, session, graph.
            if uniform_flag:
                nn_act = fp.load_pkl(load_path)
                str_dict[nn] = (nn_act, None, None)
            else:
                scope = scope_finder(load_path)
                nn_act, sess, graph = load_action_class(load_path, scope, game, training_flag=identity)
                str_dict[nn] = (nn_act, sess, graph)

    return str_dict

def find_heuristic_position(child_partition):
    """
    This function finds the starting position of different heuristics in a combined game payoff matrix.
    :param child_partition:
    :return:
    """
    position = {}
    i = 0
    for method in child_partition:
        position[method] = (i, i+child_partition[method])
        i += child_partition[method]
    return position

def create_paths(methods_list):
    """
    create paths to files that contain strategies.
    :param methods_list: a list of names of heuristics.
    :param base_path: path of the home directory.
    :return:
    """
    paths = {}
    for method in methods_list:
        paths[method] = os.getcwd() + '/combined_game/' + method
    return paths

