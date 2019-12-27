import multiprocessing as mp
import numpy as np
import random
import copy
import os
from attackgraph import file_op as fp
from attackgraph.uniform_str_init import act_def, act_att
from baselines.deepq.load_action import load_action_class

# print(os.getcwd())

def parallel_sim(env, game, nn_att, nn_def, num_episodes):

    G_list, att_list, def_list = copy_env(env, num_episodes)
    arg = list(zip(G_list,[game]*num_episodes, att_list,[nn_att]*num_episodes,def_list,[nn_def]*num_episodes,[env.T]*num_episodes))

    with mp.Pool() as pool:
        r = pool.map_async(single_sim, arg)
        a = r.get()

    return np.sum(np.array(a),0)/num_episodes


def single_sim(param): #single for single episode.
    # TODO: Dealing with uniform str
    aReward = 0
    dReward = 0
    def_uniform_flag = False
    att_uniform_flag = False

    #nn_att and nn_def here can be either np.ndarray or str. np.ndarray represents a mixed strategy.
    # A str represents the name of a strategy.
    G, game, attacker, nn_att, defender, nn_def, T = param

    if isinstance(nn_att, np.ndarray) and isinstance(nn_def, str):
        str_set = game.att_str
        nn_att = np.random.choice(str_set, p=nn_att)

    if isinstance(nn_att, str) and isinstance(nn_def, np.ndarray):
        str_set = game.def_str
        nn_def = np.random.choice(str_set, p=nn_def)

    if isinstance(nn_att, np.ndarray) and isinstance(nn_def, np.ndarray):
        str_set = game.att_str
        nn_att = np.random.choice(str_set, p=nn_att)
        str_set = game.def_str
        nn_def = np.random.choice(str_set, p=nn_def)

    if "epoch1" in nn_att:
        att_uniform_flag = True

    if "epoch1" in nn_def:
        def_uniform_flag = True

    path = os.getcwd() + "/attacker_strategies/" + nn_att
    if att_uniform_flag:
        nn_att = fp.load_pkl(path)
    else:
        training_flag = 1
        nn_att, sess1, graph1 = load_action_class(path,game,training_flag)

    path = os.getcwd() + "/defender_strategies/" + nn_def
    if def_uniform_flag:
        nn_def = fp.load_pkl(path)
    else:
        training_flag = 0
        nn_def, sess2, graph2 = load_action_class(path,game,training_flag)

    for t in range(T):
        timeleft = T - t
        if att_uniform_flag:
            attacker.att_greedy_action_builder_single(G, timeleft, nn_att)
        else:
            with graph1.as_default():
                with sess1.as_default():
                    attacker.att_greedy_action_builder_single(G, timeleft, nn_att)

        if def_uniform_flag:
            defender.def_greedy_action_builder_single(G, timeleft, nn_def)
        else:
            with graph2.as_default():
                with sess2.as_default():
                    defender.def_greedy_action_builder_single(G, timeleft, nn_def)

        att_action_set = attacker.attact
        def_action_set = defender.defact
        # print('att:', att_action_set)
        # print('def:', def_action_set)
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
        _, targetset = get_Targets(G)
        for node in targetset:
            if G.nodes[node]['state'] == 1:
                aReward += G.nodes[node]['aReward']
                dReward += G.nodes[node]['dPenalty']

    # print(aReward)
    # print(aReward, dReward)
    return aReward, dReward

def get_Targets(G):
    count = 0
    targetset = set()
    for node in G.nodes:
        if G.nodes[node]['type'] == 1:
            count += 1
            targetset.add(node)
    return count,targetset

def copy_env(env, num_episodes):
    G_list = []
    att_list = []
    def_list = []
    env.reset_everything()
    for _ in np.arange(num_episodes):
        G_list.append(copy.deepcopy(env.G_reserved))
        att_list.append(copy.deepcopy(env.attacker))
        def_list.append(copy.deepcopy(env.defender))

    return G_list, att_list, def_list
