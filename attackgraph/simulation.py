import random
import numpy as np
from baselines.deepq.load_action import load_action_class
import copy
import os
from attackgraph import file_op as fp


def series_sim(env, game, nn_att, nn_def, num_episodes):
    aReward_list = np.array([])
    dReward_list = np.array([])
    nn_att_saved = copy.copy(nn_att)
    nn_def_saved = copy.copy(nn_def)

    T = env.T

    # Test if nn_att and nn_def point to one single strategy.
    single_str_att = True
    single_str_def = True
    if isinstance(nn_att, np.ndarray):
        if len(np.where(nn_att>0.95)[0]) != 1:
            single_str_att = False

    if isinstance(nn_def, np.ndarray):
        if len(np.where(nn_def>0.95)[0]) != 1:
            single_str_def = False

    _, targetset = get_Targets(env.G)

    for i in range(num_episodes): #can be run parallel

        # G = copy.deepcopy(env.G_reserved)
        # attacker = copy.deepcopy(env.attacker)
        # defender = copy.deepcopy(env.defender)

        env.reset_everything()
        G = env.G
        attacker = env.attacker
        defender = env.defender


        aReward = 0
        dReward = 0

        if i == 0 or not single_str_att:
            att_uniform_flag = False
            nn_att = copy.copy(nn_att_saved)
            if isinstance(nn_att, np.ndarray):
                str_set = game.att_str
                nn_att = np.random.choice(str_set, p=nn_att)

            if "epoch1.pkl" in nn_att:
                att_uniform_flag = True

            path = os.getcwd() + "/attacker_strategies/" + nn_att
            if att_uniform_flag:
                nn_att_act = fp.load_pkl(path)
            else:
                training_flag = 1
                nn_att_act, sess1, graph1 = load_action_class(path, nn_att, game, training_flag)

        if i == 0 or not single_str_def:
            def_uniform_flag = False
            nn_def = copy.copy(nn_def_saved)
            if isinstance(nn_def, np.ndarray):
                str_set = game.def_str
                nn_def = np.random.choice(str_set, p=nn_def)

            if "epoch1.pkl" in nn_def:
                def_uniform_flag = True

            path = os.getcwd() + "/defender_strategies/" + nn_def
            if def_uniform_flag:
                nn_def_act = fp.load_pkl(path)
            else:
                training_flag = 0
                nn_def_act, sess2, graph2 = load_action_class(path, nn_def, game, training_flag)




        # def_uniform_flag = False
        # att_uniform_flag = False
        #
        # nn_att = copy.copy(nn_att_saved)
        # nn_def = copy.copy(nn_def_saved)
        #
        # # nn_att and nn_def here can be either np.ndarray or str. np.ndarray represents a mixed strategy.
        # # A str represents the name of a strategy.
        #
        # if isinstance(nn_att, np.ndarray) and isinstance(nn_def, str):
        #     str_set = game.att_str
        #     nn_att = np.random.choice(str_set, p=nn_att)
        #
        # if isinstance(nn_att, str) and isinstance(nn_def, np.ndarray):
        #     str_set = game.def_str
        #     nn_def = np.random.choice(str_set, p=nn_def)
        #
        # if isinstance(nn_att, np.ndarray) and isinstance(nn_def, np.ndarray):
        #     str_set = game.att_str
        #     nn_att = np.random.choice(str_set, p=nn_att)
        #     str_set = game.def_str
        #     nn_def = np.random.choice(str_set, p=nn_def)
        #
        # if "epoch1" in nn_att:
        #     att_uniform_flag = True
        #
        # if "epoch1" in nn_def:
        #     def_uniform_flag = True
        #
        # path = os.getcwd() + "/attacker_strategies/" + nn_att
        # if att_uniform_flag:
        #     nn_att_act = fp.load_pkl(path)
        # else:
        #     training_flag = 1
        #     nn_att_act, sess1, graph1 = load_action_class(path, nn_att, game, training_flag)
        #
        # path = os.getcwd() + "/defender_strategies/" + nn_def
        # if def_uniform_flag:
        #     nn_def_act = fp.load_pkl(path)
        # else:
        #     training_flag = 0
        #     nn_def_act, sess2, graph2 = load_action_class(path, nn_def, game, training_flag)





        # print('===================================')
        # print('==========start episode============')
        # print('===================================')
        # print(aReward, dReward)

        for t in range(T):
            # print('====================')
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
            # print(t, 'att:', att_action_set)
            # print(t, 'def:', def_action_set)
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

            # print('Before Traget aRew:', aReward, 'dRew:', dReward)
            # print('target set:', targetset)
            # current_state = []
            # for node in G.nodes:
            #     current_state.append(G.nodes[node]['state'])
            # print('current_state:', current_state)
            for node in targetset:
                if G.nodes[node]['state'] == 1:
                    aReward += G.nodes[node]['aReward']
                    dReward += G.nodes[node]['dPenalty']
            # print('aRew:', aReward, 'dRew:', dReward)

            # update players' observations
            # update defender's observation
            defender.update_obs(defender.get_def_hadAlert(G))
            defender.save_defact2prev()
            defender.defact.clear()
            # update attacker's observation
            attacker.update_obs(attacker.get_att_isActive(G))
            attacker.attact.clear()

        aReward_list = np.append(aReward_list,aReward)
        dReward_list = np.append(dReward_list,dReward)
        # print('alist:', aReward_list)
        # print('dlist:', dReward_list)

    return np.round(np.mean(aReward_list),2), np.round(np.mean(dReward_list),2)

def get_Targets(G):
    count = 0
    targetset = set()
    for node in G.nodes:
        if G.nodes[node]['type'] == 1:
            count += 1
            targetset.add(node)
    return count,targetset