import numpy as np
from attackgraph.utils_combined import find_heuristic_position
from attackgraph.gambit_analysis import do_gambit_analysis
from attackgraph.cournot import extract_submatrix as es
from attackgraph import file_op as fp
from queue import PriorityQueue as pq
import os
import random

def eligibility_trace(nasheq_dict, child_partition, gamma=0.8):
    '''
    This function calculates the eligibility trace of strategies based on game.nasheq.
    :param nasheq_dict: {"baselines": game.nasheq}
    :param gamma:
    :return:
    '''
    position = find_heuristic_position(child_partition)
    total_num_str = 0
    for method in child_partition:
        total_num_str += child_partition[method]

    et_dict_def = {}
    et_dict_att = {}
    for method in nasheq_dict:
        et_dict_def[method] = np.zeros(child_partition[method])
        et_dict_att[method] = np.zeros(child_partition[method])
        nasheq = nasheq_dict[method]
        for epoch in nasheq:
            et_dict_att[method] *= gamma
            ne_att = nasheq[epoch][1][1:]
            ne_att[ne_att > 0] = 1
            et_dict_att[method][:len(ne_att)] += ne_att

            et_dict_def[method] *= gamma
            ne_def = nasheq[epoch][0][1:]
            ne_def[ne_def>0] = 1
            et_dict_def[method][:len(ne_def)] += ne_def

    pq_def = {}
    pq_att = {}
    for method in et_dict_def:
        pq_def[method] = pq()
        pq_att[method] = pq()
        start, end = position[method]
        idx_str = start + np.arange(child_partition[method])
        idx_et_pair_def = zip(idx_str, et_dict_def[method])
        idx_et_pair_att = zip(idx_str, et_dict_att[method])
        for pair in idx_et_pair_def:
            pq_def[method].put(pair)

        for pair in idx_et_pair_att:
            pq_att[method].put(pair)

    return pq_def, pq_att


def create_ne_dict(method_list):
    load_path = os.getcwd() + '/combined_game/games/'

    ne_dict = {}
    for method in method_list:
        path = load_path + method + '/game.pkl'
        game = fp.load_pkl(path)
        ne_dict[method] = game.nasheq

    return ne_dict

def ne_search_wo_et(payoff_matrix_def, payoff_matrix_att, child_partition, ne_dict, pq_def=None, pq_att=None):
    position = find_heuristic_position(child_partition)

    total_num_str = 0
    init_flag = False

    # Assume 2 methods.
    for method in ne_dict:
        if not init_flag:
            candidate_def = ne_dict[method][0][1:]
            candidate_att = ne_dict[method][1][1:]
            init_flag = True

        total_num_str += child_partition[method]


    # Extend the NE to the length of the combined game.
    zeros = np.zeros(total_num_str)
    candidate_def = zeros[:len(candidate_def)] + candidate_def
    candidate_att = zeros[:len(candidate_att)] + candidate_att

    # indicator_matrix records which cell has been simulated in the payoff matrix.
    indicator_matrix = np.zeros((total_num_str, total_num_str))
    for method in position:
        start, end = position[method]
        indicator_matrix[start:end, start:end] = 1

    candidate_def = np.reshape(candidate_def, newshape=(len(candidate_def), 1))

    payoff_def = np.sum(candidate_def * payoff_matrix_def * candidate_att)
    payoff_att = np.sum(candidate_def * payoff_matrix_att * candidate_att)

    support_idx_def = np.where(candidate_def>0)[0]
    support_idx_att = np.where(candidate_att>0)[0]

    for x in support_idx_def:
        for y in support_idx_att:
            indicator_matrix[x, y] = 5

    # Sampling scheme could be changed.
    if pq_def is None:
        sample_set_def = set(np.arange(total_num_str)) - set(support_idx_def)
        sample_set_att = set(np.arange(total_num_str)) - set(support_idx_att)

    confirmed_flag_def = False
    confirmed_flag_att = False

    if pq_def is None:
        dev_def = random.sample(sample_set_def, 1)
        dev_att = random.sample(sample_set_att, 1)
    else:
        # TODO: pq_def is a dict
        dev_def, _ = pq_def.get()
        dev_att, _ = pq_att.get()

    # Change to simulation mode when simulation is needed.
    while not confirmed_flag_def or not confirmed_flag_att:

        for x in support_idx_def:
            indicator_matrix[x, dev_att] = 1
        for y in support_idx_att:
            indicator_matrix[dev_def, y] = 1

        dev_payoff_def = np.sum(payoff_matrix_def[dev_def, :] * candidate_att)
        dev_payoff_att = np.sum(candidate_def * payoff_matrix_att[:, dev_att])

        # TODO: add pq() sampling.
        if dev_payoff_def > payoff_def:
            support_idx_def = np.append(support_idx_def, dev_def)
        if dev_payoff_att > payoff_att:
            support_idx_att = np.append(support_idx_att, dev_att)

        subgame_def = es(support_idx_def, support_idx_att, payoff_matrix_def)
        subgame_att = es(support_idx_def, support_idx_att, payoff_matrix_att)

        nash_att, nash_def = do_gambit_analysis(subgame_def, subgame_att, maxent=False, minent=False)
        nash_def = np.reshape(nash_def, newshape=(len(nash_def), 1))

        payoff_def = np.sum(nash_def * payoff_matrix_def * nash_att)
        payoff_att = np.sum(nash_def * payoff_matrix_att * nash_att)

        nash_def = np.reshape(nash_def, newshape=np.shape(nash_att))

        candidate_def = np.zeros(total_num_str)
        candidate_att = np.zeros(total_num_str)
        for idx, value in zip(support_idx_def, nash_def):
            candidate_def[idx] = value
        for idx, value in zip(support_idx_att, nash_att):
            candidate_att[idx] = value

        if len(set(np.arange(total_num_str)) - set(support_idx_def)) == 0:
            confirmed_flag_def = True
        else:
            sample_set_def = set(np.arange(total_num_str)) - set(support_idx_def)
            dev_def = random.sample(sample_set_def, 1)

        if len(set(np.arange(total_num_str)) - set(support_idx_att)) == 0:
            confirmed_flag_att = True
        else:
            sample_set_att = set(np.arange(total_num_str)) - set(support_idx_att)
            dev_att = random.sample(sample_set_att, 1)


    return candidate_def, candidate_att, indicator_matrix




























