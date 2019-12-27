"""
This function is designed for computing the NE of combined game based on the
inner loop NE search algorithm of EGTA.

This is a tensorflow version without constrained by the scope of variables.
"""
import numpy as np
from attackgraph import file_op as fp
from attackgraph import json_op as jp
import os
import warnings
from attackgraph.gambit_analysis import do_gambit_analysis
from attackgraph.deepgraph_runner import initialize
from attackgraph.simulation import get_Targets
from baselines.deepq.load_action import load_action_class
import random
import sys
import joblib
import copy
from queue import PriorityQueue
# sys.path.append('/home/wangyzh/combined')

str_path_att = os.getcwd() + "/combined_game/attacker/"
str_path_def = os.getcwd() + "/combined_game/defender/"

"""
The rule of creating subgames of the combined game:
1. The first subgame is created by only including the NE strategies in each child game.
2. If there is any benefitial deviation, then extend the subgame by adding strategies according to their
   frequency in each of the child game.
"""

def subgame_ne_search(num_str, main_method, nash_eq, payoff_matrice, child_partition):
    # initialize the game.
    game = fp.load_pkl(os.getcwd() + '/game_data/game.pkl')

    co_payoff_matrix_def, co_payoff_matrix_att, deviations_def, deviations_att = init_simulation(num_str,
                                                                                                 main_method,
                                                                                                 nash_eq,
                                                                                                 payoff_matrice)

    # NE in the main_method.
    # Assumption: both players have same number of strategies.
    nasheq = nash_eq[main_method]
    mixed_str_def = np.zeros(num_str)
    mixed_str_att = np.zeros(num_str)
    mixed_str_def[:len(nasheq[0])] = nasheq[0]
    mixed_str_att[:len(nasheq[1])]=  nasheq[1]

    # Construct defender's/attacker's strategy lists.
    str_list_def = []
    str_list_att = []

    for method in child_partition:
        for i in np.arange(2,2+child_partition[method]):
            str_list_def.append(method + "_def_str_epoch" + str(i) + '.pkl')
            str_list_att.append(method + "_att_str_epoch" + str(i) + '.pkl')

    # Set flags.
    confirmed = False
    def_dev_flag = False
    att_dev_flag = False

    # initialize strategy library.
    str_dict_def = load_policies(game, child_partition, identity=0)
    str_dict_att = load_policies(game, child_partition, identity=1)

    # NE Payoff for both players.
    _mixed_str_def = np.reshape(mixed_str_def,newshape=(num_str,1))
    ne_util_def = np.sum(_mixed_str_def * co_payoff_matrix_def * mixed_str_def)
    ne_util_att = np.sum(_mixed_str_def * co_payoff_matrix_att * mixed_str_att)

    # Search for defender's deviation.
    ne_pos_def = np.where(mixed_str_def>0)[0]
    ne_pos_att = np.where(mixed_str_att>0)[0]

    cur_dev_def = deviations_def.pop()
    idx_def = np.where(str_list_def == cur_dev_def)[0]
    if len(idx_def) == 0:
        raise ValueError("Current deviation strategy does not exist.")
    for att in ne_pos_att:
        aReward, dReward = series_sim_combined(env, game, str_dict_att[str_list_att[att]], str_dict_def[cur_dev_def],game.num_episodes)
        co_payoff_matrix_att[idx_def[0], att] = aReward
        co_payoff_matrix_def[idx_def[0], att] = dReward


    # Search for attacker's deviation.



def init_simulation(num_str,
                    main_method,
                    child_partition,
                    nash_eq,
                    payoff_matrice,
                    ne_memory = 5):
    """
    This function initializes the subgame 0 and outputs the ordering of potential deviations. 2 players assumed.
    :param num_str: number of strategies in total
    :param main_method: the method that initilizes game 0.
    :param child_partition: a dict recording number of strategies for each child game. {"baseline": 40, "RS":40}
    :param nash_eq: a dict recording the NE of each method {"baseline": nasheq, "RS": nasheq, ...}
    :param payoff_matrice: a dict recording the payoff matrix of each method {"baseline": payoff, "RS": payoff, ...}
                           payoff[0]/[1] def/att payoff matrix. Uniform strategy removed.
    :return:
    """

    # ranking potential beneficial deviation.
    methods = list(child_partition.keys)
    print("The methods for this simulation include ", methods)

    # Deviation only considers strategies of other methods.
    deviations_att = PriorityQueue()
    deviations_def = PriorityQueue()

    if main_method not in child_partition:
        raise ValueError("main_method is not a key.")

    for method in child_partition:
        if method == main_method:
            continue
        ranking_ne(method, nash_eq[method], deviations_att, deviations_def, ne_memory)


    # Initialize Payoff matrix of combined game
    # payoff[0] = payoffmatrix_def
    # payoff[0] = payoffmatrix_att

    count = 0
    for method in child_partition:
        count += child_partition[method]
    if count != num_str:
        raise ValueError("The number of strategies is not equal to str_num in child_partition.")

    co_payoff_matrix_def = np.zeros((num_str, num_str))
    co_payoff_matrix_att = np.zeros((num_str, num_str))
    position = 0
    for method in payoff_matrice:
        co_payoff_matrix_def[position:position+child_partition[method], position:position+child_partition[method]] = payoff_matrice[method][0]
        co_payoff_matrix_att[position:position + child_partition[method], position:position + child_partition[method]] = payoff_matrice[method][1]
        position += child_partition[method]


    return co_payoff_matrix_def, co_payoff_matrix_att, deviations_def, deviations_att


#TODO: remove the name rule (+method).
def ranking_ne(method, nasheq, deviation_att, deviation_def, ne_memory):
    """
    ranking_ne assigns highest priority to NE strategies in other methods, then strategies with high NE frequency,
    then regular strategies.
    Priority Structure:
    0: NE strategies.
    1: The strategy with highest frequency outside the support.
    ...
    999: intermediate strategies
    :param nasheq: game.nasheq
    :param deviation: PriorityQueue
    :return:
    """
    num_epoch = len(nasheq.keys())

    if num_epoch <= ne_memory:
        raise ValueError("The number of num_epoch is less the ne_memory.")

    # ordering defender's deviation
    num_str =  len(nasheq[0][0])
    frequency_att = np.zeros(num_str)
    frequency_def = np.zeros(num_str)
    for i in np.arange(ne_memory):
        epoch = num_epoch - i
        ne = nasheq[epoch][0]
        ne_str = np.where(ne>0)[0]
        for idx in ne.str:
            str_name = method + '_def_str_epoch' + str(idx+1) + '.pkl'
            if i == 0:
                deviation_def.put((0,str_name))
                ne_def_copy = ne_str.copy()
            frequency_def[idx] += 1

    # ordering attacker's deviation
    for i in np.arange(ne_memory):
        epoch = num_epoch - i
        ne = nasheq[epoch][1]
        ne_str = np.where(ne > 0)[0]
        for idx in ne.str:
            str_name = method + '_att_str_epoch' + str(idx + 1) + '.pkl'
            if i == 0:
                deviation_att.put((0, str_name))
                ne_att_copy = ne_str.copy()
            frequency_att[idx] += 1

    frequency_idx_def = np.flip(np.argsort(frequency_def))
    frequency_idx_att = np.flip(np.argsort(frequency_att))

    #TODO: make sure this is correct.
    for i in frequency_idx_att:
        if i not in ne_att_copy:
            deviation_att.put((i+1, method + '_att_str_epoch' + str(i + 1) + '.pkl'))
        else:
            continue

    for i in frequency_idx_def:
        if i not in ne_def_copy:
            deviation_def.put((i+1, method + '_def_str_epoch' + str(i + 1) + '.pkl'))
        else:
            continue

    # return deviation_att, deviation_def



"""
The code below simulate the complete payoff matrix.
"""

def whole_payoff_matrix(num_str, child_partition, env_name='run_env_B', save_path=None, str_path_dict=None, matrix_name=None):
    """
    Simulate the complete payoff matrix.
    :param num_str: number of total strategies.
    :param child_partition: a dict recording number of strategies for each child game. {"baseline": 40, "RS":40}
    :param env_name: name of envrionment
    :param str_range: the range of strategies to be simulated. {"start": 3, "end": 20}
    :param str_path_dict: {0/1:{'baselines': '/home/wangyzh/baselines/attackgraph/attacker_strategies/'}}
    :return: NE
    """
    print('Begin simulating payoff matrix of combined game.')
    game = initialize(load_env=env_name, env_name=None)

    env = game.env
    num_episodes = game.num_episodes

    # Assume two players have the same number of strategies.
    payoff_matrix_att = np.zeros((num_str, num_str))
    payoff_matrix_def = np.zeros((num_str, num_str))

    #TODO: check the load path.
    att_str_dict = load_policies(game, child_partition, identity=0, path_dict=str_path_dict)
    def_str_dict = load_policies(game, child_partition, identity=1, path_dict=str_path_dict)

    # method_pos_def records the starting idx of each method when combined.
    method_pos_def = 0
    for key_def in child_partition:
        for i in np.arange(1, child_partition[key_def]+1):
            def_str = key_def + '_def_str_epoch' + str(i+1) + '.pkl'
            entry_pos_def = method_pos_def + i
            method_pos_att = 0
            for key_att in child_partition:
                for j in np.arange(1,child_partition[key_att]+1):
                    att_str = key_att + '_att_str_epoch' + str(j+1) + '.pkl'
                    entry_pos_att = method_pos_att + j
                    # print current simulation info.
                    if j % 10 == 0:
                        print('Current Method is ', key_def, key_att, 'Current position:', i,j)
                        sys.stdout.flush()

                    att_nn = att_str_dict[att_str]
                    def_nn = def_str_dict[def_str]

                    aReward, dReward = series_sim_combined(env, att_nn, def_nn, num_episodes)

                    payoff_matrix_att[entry_pos_def-1, entry_pos_att-1] = aReward
                    payoff_matrix_def[entry_pos_def-1, entry_pos_att-1] = dReward

                # update the starting position.
                method_pos_att += child_partition[key_att]

        # Periodically saving the payoff matrix.
        if save_path is None:
            save_path = os.getcwd() + '/combined_game/matrice/'
        if matrix_name is None:
            fp.save_pkl(payoff_matrix_att, save_path + 'payoff_matrix_att.pkl')
            fp.save_pkl(payoff_matrix_def, save_path + 'payoff_matrix_def.pkl')
        else:
            fp.save_pkl(payoff_matrix_att, save_path + 'payoff_matrix_att_' + matrix_name + '.pkl')
            fp.save_pkl(payoff_matrix_def, save_path + 'payoff_matrix_def_' + matrix_name + '.pkl')
        method_pos_def += child_partition[key_def]

    print('Done simulating payoff matrix of combined game.')
    return payoff_matrix_att, payoff_matrix_def


def regret(nash_att, nash_def, payoffmatrix_att, payoffmatrix_def):
    num_str = len(nash_att)
    x1, y1 = np.shape(payoffmatrix_def)
    x2, y2 = np.shape(payoffmatrix_att)
    if x1 != y1 or x1 != x2 or x2 != y2 or x1 != num_str:
        raise ValueError("Dim of NE does not match payoff matrix.")

    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.sum(nash_def * payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * payoffmatrix_att * nash_att), decimals=2)

    utils_def = np.round(np.sum(payoffmatrix_def * nash_att, axis=1), decimals=2)
    utils_att = np.round(np.sum(nash_def * payoffmatrix_att, axis=0), decimals=2)

    regret_def = utils_def - dPayoff
    regret_att = utils_att - aPayoff

    regret_def = np.reshape(regret_def, newshape=np.shape(regret_att))

    regret_att = -regret_att
    regret_def = -regret_def

    return regret_att, regret_def


#TODO: change the split point according to child_partition. Mean regret does not make sense.
def mean_regret(regret_att, regret_def, child_partition):
    mean_reg_att = []
    mean_reg_def = []
    mean_reg_att.append(np.round(np.mean(regret_att[1:41]), decimals=2))
    mean_reg_def.append(np.round(np.mean(regret_def[1:41]), decimals=2))
    mean_reg_att.append(np.round(np.mean(regret_att[42:83]), decimals=2))
    mean_reg_def.append(np.round(np.mean(regret_def[42:83]), decimals=2))
    mean_reg_att.append(np.round(np.mean(regret_att[84:124]), decimals=2))
    mean_reg_def.append(np.round(np.mean(regret_def[84:124]), decimals=2))
    return mean_reg_att, mean_reg_def


def NE_regret(regret_att, regret_def, ne_dict):
    """
    Calculate the regret of each heuristic with respect to the combined game. The strategies of each heuristic only\
    include those in the NE of each heuristic.
    :param regret_att: attacker's regret vector calculated from combined game.
    :param regret_def: defender's regret vector calculated from combined game.
    :param ne_dict: {"baseline": np.array([1,0,1,0...]), "RS": np.array([0,0,1,0...])} when a strategy is in a NE, that strategy is indicated by 1.
    :return:
    """
    num_method = len(ne_dict.keys())
    for method in ne_dict:
        pass




# Measure the regret of subgames during strategy exploration.
def regret_curve():
    pass


# Replaceability Implementation.
def replaceability():
    pass


def series_sim_combined(env, nn_att, nn_def, num_episodes):
    aReward_list = np.array([])
    dReward_list = np.array([])

    T = env.T

    _, targetset = get_Targets(env.G)

    for i in range(num_episodes): #can be run parallel

        env.reset_everything()
        G = env.G
        attacker = env.attacker
        defender = env.defender

        aReward = 0
        dReward = 0

        att_uniform_flag = False
        def_uniform_flag = False

        nn_att_act, sess1, graph1 = nn_att
        nn_def_act, sess2, graph2 = nn_def

        if sess1 == None:
            att_uniform_flag = True

        if sess2 == None:
            def_uniform_flag = True

        for t in range(T):
            timeleft = T - t
            if att_uniform_flag:
                attacker.att_greedy_action_builder_single(G, timeleft, nn_att_act)
            else:
                with graph1.as_default():
                    with sess1.as_default():
                        attacker.att_greedy_action_builder_single(G, timeleft, nn_att_act)

            if def_uniform_flag:
                defender.def_greedy_action_builder_single(G, timeleft, nn_def_act)
            else:
                with graph2.as_default():
                    with sess2.as_default():
                        defender.def_greedy_action_builder_single(G, timeleft, nn_def_act)

            att_action_set = attacker.attact
            def_action_set = defender.defact

            for attack in att_action_set:
                if isinstance(attack, tuple):
                    # check OR node
                    aReward += G.edges[attack]['cost']
                    if random.uniform(0, 1) <= G.edges[attack]['actProb']:
                        G.nodes[attack[-1]]['state'] = 1
                else:
                    # check AND node
                    aReward += G.nodes[attack]['aCost']
                    if random.uniform(0, 1) <= G.nodes[attack]['actProb']:
                        G.nodes[attack]['state'] = 1
            # defender's action
            for node in def_action_set:
                G.nodes[node]['state'] = 0
                dReward += G.nodes[node]['dCost']

            for node in targetset:
                if G.nodes[node]['state'] == 1:
                    aReward += G.nodes[node]['aReward']
                    dReward += G.nodes[node]['dPenalty']

            #update players' observations
            #update defender's observation
            defender.update_obs(defender.get_def_hadAlert(G))
            defender.save_defact2prev()
            defender.defact.clear()
            #update attacker's observation
            attacker.update_obs(attacker.get_att_isActive(G))
            attacker.attact.clear()

        aReward_list = np.append(aReward_list,aReward)
        dReward_list = np.append(dReward_list,dReward)

    return np.round(np.mean(aReward_list),2), np.round(np.mean(dReward_list),2)

def scope_finder(policy_path):
    if not fp.isExist(policy_path):
        raise ValueError("Policy does not exist.")
    loaded_params = joblib.load(os.path.expanduser(policy_path))
    scope = iter(loaded_params).__next__().split('/')[0]
    return scope

# Load all policies into a dictionary.
def load_policies(game, child_partition, identity, path_dict=None):
    if identity == 0: # load defender's policies.
        mid_name = '_def_str_epoch'
        if path_dict is None:
            path = str_path_def
    elif identity == 1:
        mid_name = '_att_str_epoch'
        if path_dict is None:
            path = str_path_att
    else:
        raise ValueError("identity is not correct")

    str_dict = {}
    if path_dict is None:
        for key in child_partition:
            for i in np.arange(1, child_partition[key]+1):
                nn = key + mid_name + str(i+1) + '.pkl'

                uniform_flag = False
                if "epoch1.pkl" in nn:
                    uniform_flag = True

                load_path = path + nn

                # Strategies are kept as a tuple with parameters, session, graph.
                if uniform_flag:
                    nn_act = fp.load_pkl(load_path)
                    str_dict[nn] = (nn_act, None, None)
                else:
                    scope = scope_finder(path)
                    nn_act, sess, graph = load_action_class(load_path, scope, game, training_flag=identity)
                    str_dict[nn] = (nn_act, sess, graph)
    else:
        for key in child_partition:
            str_dict[key] = {}
            for i in np.arange(1, child_partition[key] + 1):
                nn = key + mid_name + str(i + 1) + '.pkl'

                uniform_flag = False
                if "epoch1.pkl" in nn:
                    uniform_flag = True


                path = path_dict[identity][key]
                load_path = path + nn

                # Strategies are kept as a tuple with parameters, session, graph.
                if uniform_flag:
                    nn_act = fp.load_pkl(load_path)
                    str_dict[key][nn] = (nn_act, None, None)
                else:
                    scope = scope_finder(path)
                    nn_act, sess, graph = load_action_class(load_path, scope, game, training_flag=identity)
                    str_dict[key][nn] = (nn_act, sess, graph)

    return str_dict