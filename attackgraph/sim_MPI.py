from mpi4py import MPI
from baselines.deepq.load_action import load_action_class
from attackgraph import file_op as fp
from attackgraph.subproc import call_and_wait
import os
import numpy as np
import random
import copy

def do_MPI_sim(nn_att, nn_def):
    path = os.getcwd()
    path_att = path + '/sim_arg/nn_att.pkl'
    path_def = path + '/sim_arg/nn_def.pkl'
    fp.save_pkl(nn_att, path_att)
    fp.save_pkl(nn_def, path_def)

    command_line = "mpirun python " + path + "/sim_MPI.py"

    # aReward_list = []
    # dReward_list = []
    # num_mpirun = 5
    # for i in range(num_mpirun):
    #     call_and_wait(command_line)
    #     aReward, dReward = fp.load_pkl(path + '/sim_arg/result.pkl')
    #     aReward_list.append(aReward)
    #     dReward_list.append(dReward)

    call_and_wait(command_line)
    # sim_and_modifiy_MPI()
    aReward, dReward = fp.load_pkl(path + '/sim_arg/result.pkl')

    return aReward, dReward
    # return np.sum(aReward_list)/num_mpirun, np.sum(dReward_list)/num_mpirun


#TODO: Is the game saving enough info.
def sim_and_modifiy_MPI():

    path = os.getcwd()
    game_path = os.getcwd() + '/game_data/game.pkl'
    game = fp.load_pkl(game_path)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size


    nn_att = fp.load_pkl(path + '/sim_arg/nn_att.pkl')
    nn_def = fp.load_pkl(path + '/sim_arg/nn_def.pkl')

    aReward, dReward = series_sim(game.env, game, nn_att, nn_def, size)
    # aReward, dReward = series_sim_single(game.env, game, nn_att, nn_def)
    reward_tuple = (aReward, dReward)
    data = comm.gather(reward_tuple, root = 0)
    if rank == 0:
        data = np.array(data)
        fp.save_pkl(np.round(np.sum(data,0)/size, 1), path = path + '/sim_arg/result.pkl')


def series_sim(env, game, nn_att, nn_def, size):
    aReward_list = np.array([])
    dReward_list = np.array([])

    nn_att_saved = copy.copy(nn_att)
    nn_def_saved = copy.copy(nn_def)

    if size > 20:
        num_epi = 10
    elif size >10 and size <= 20:
        num_epi = 20
    else:
        num_epi = 30

    for i in range(2):
        G = copy.deepcopy(env.G_reserved)
        attacker = copy.deepcopy(env.attacker)
        defender = copy.deepcopy(env.defender)
        T = env.T

        aReward = 0
        dReward = 0
        def_uniform_flag = False
        att_uniform_flag = False

        nn_att = copy.copy(nn_att_saved)
        nn_def = copy.copy(nn_def_saved)

        # nn_att and nn_def here can be either np.ndarray or str. np.ndarray represents a mixed strategy.
        # A str represents the name of a strategy.

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
            nn_att_act = fp.load_pkl(path)
        else:
            training_flag = 1
            nn_att_act, sess1, graph1 = load_action_class(path, nn_att, game, training_flag)

        path = os.getcwd() + "/defender_strategies/" + nn_def
        if def_uniform_flag:
            nn_def_act = fp.load_pkl(path)
        else:
            training_flag = 0
            nn_def_act, sess2, graph2 = load_action_class(path, nn_def, game, training_flag)

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

        aReward_list = np.append(aReward_list,aReward)
        dReward_list = np.append(dReward_list,dReward)

    return np.mean(aReward_list), np.mean(dReward_list)


def series_sim_single(env, game, nn_att, nn_def):

    G = copy.deepcopy(env.G_reserved)
    attacker = copy.deepcopy(env.attacker) #TODO: reset att? maybe no.
    defender = copy.deepcopy(env.defender)
    T = env.T

    aReward = 0
    dReward = 0
    def_uniform_flag = False
    att_uniform_flag = False

    # nn_att and nn_def here can be either np.ndarray or str. np.ndarray represents a mixed strategy.
    # A str represents the name of a strategy.

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
        nn_att_act = fp.load_pkl(path)
    else:
        training_flag = 1
        nn_att_act, sess1, graph1 = load_action_class(path, nn_att, game, training_flag)

    path = os.getcwd() + "/defender_strategies/" + nn_def
    if def_uniform_flag:
        nn_def_act = fp.load_pkl(path)
    else:
        training_flag = 0
        nn_def_act, sess2, graph2 = load_action_class(path, nn_def, game, training_flag)

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

    return np.round(aReward,2), np.round(dReward,2)

def get_Targets(G):
    count = 0
    targetset = set()
    for node in G.nodes:
        if G.nodes[node]['type'] == 1:
            count += 1
            targetset.add(node)
    return count,targetset



if __name__ == '__main__':
    sim_and_modifiy_MPI()