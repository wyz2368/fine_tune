import numpy as np
from attackgraph import file_op as fp
import os
from attackgraph.deepgraph_runner import initialize
from attackgraph.simulation import get_Targets
from attackgraph.utils_combined import load_policies, screen, create_paths
# from baselines.deepq.load_action import load_action_class
import random
import sys


def whole_payoff_matrix_restart(num_str,
                                child_partition,
                                done_list,
                                payoff_matrix_att,
                                payoff_matrix_def,
                                env_name='run_env_B',
                                save_path=None,
                                matrix_name=None):
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
    print("*********************************************")
    print("*********************************************")
    game = initialize(load_env=env_name, env_name=None)
    print("*********************************************")
    print("*********************************************")
    sys.stdout.flush()

    env = game.env
    num_episodes = game.num_episodes

    # Assume two players have the same number of strategies.
    # payoff_matrix_att = np.zeros((num_str, num_str))
    # payoff_matrix_def = np.zeros((num_str, num_str))

    att_str_dict = load_policies(game, child_partition, identity=1)
    def_str_dict = load_policies(game, child_partition, identity=0)

    ## method_pos_def records the starting idx of each method when combined.
    method_pos_def = 0
    for key_def in child_partition:
        if key_def in done_list:
            method_pos_def += child_partition[key_def]
            continue
        for i in np.arange(1, child_partition[key_def]+1):
            def_str = key_def + '/defender_strategies/def_str_epoch' + str(i+1) + '.pkl'
            entry_pos_def = method_pos_def + i
            method_pos_att = 0
            for key_att in child_partition:
                print('Current Method is ', (key_def, key_att), "Defender's pos is ", i+1, '# attacker strategies is ', child_partition[key_att])
                sys.stdout.flush()
                for j in np.arange(1,child_partition[key_att]+1):
                    att_str = key_att + '/attacker_strategies/att_str_epoch' + str(j+1) + '.pkl'
                    entry_pos_att = method_pos_att + j
                    # print current simulation info.
                    # if j == child_partition[key_att]:
                    #     print("----------------------------------------------------")
                    #     print('Current position:', (i+1,j+1), 'Pos:', (entry_pos_def-1, entry_pos_att-1))
                    #     sys.stdout.flush()

                    att_nn = att_str_dict[att_str]
                    def_nn = def_str_dict[def_str]

                    aReward, dReward = series_sim_combined(env, att_nn, def_nn, num_episodes=num_episodes)

                    payoff_matrix_att[entry_pos_def-1, entry_pos_att-1] = aReward
                    payoff_matrix_def[entry_pos_def-1, entry_pos_att-1] = dReward

                # update the starting position.
                method_pos_att += child_partition[key_att]

        ## Periodically saving the payoff matrix.
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


def scan_and_sim_restart(methods_list, done_list):
    paths = create_paths(methods_list)
    str_dict_def, str_dict_att = screen(paths)

    # create children partition
    child_partition = {}
    total_num_str = 0
    for method in methods_list:
        child_partition[method] = len(str_dict_def[method])
        print("Method is ", method, "which has " , child_partition[method], " strategies.")
        total_num_str += child_partition[method]

    load_path = os.getcwd() + '/combined_game/matrice/'
    payoff_matrix_att = fp.load_pkl(load_path + 'payoff_matrix_att.pkl')
    payoff_matrix_def = fp.load_pkl(load_path + 'payoff_matrix_def.pkl')


    payoff_matrix_att, payoff_matrix_def = whole_payoff_matrix_restart(total_num_str,
                                                                       child_partition,
                                                                       done_list,
                                                                       payoff_matrix_att,
                                                                       payoff_matrix_def)

    return payoff_matrix_att, payoff_matrix_def, child_partition