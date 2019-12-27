# Packages import
import numpy as np
import os
import datetime
import sys
import psutil
import warnings

# Modules import
from attackgraph import DagGenerator as dag
from attackgraph import file_op as fp
from attackgraph import json_op as jp
from attackgraph import sim_Series
from attackgraph import training
from attackgraph import util
from attackgraph import game_data
from attackgraph import gambit_analysis as ga
from attackgraph.simulation import series_sim
# from attackgraph.sim_MPI import do_MPI_sim
from attackgraph.sim_retrain import sim_retrain
from attackgraph.egta_subproc import do_train_and_sim
# from attackgraph.subproc import call_and_wait


# def do_train_and_sim():
#     print('Begin do_train_and_sim')
#     path = os.getcwd()
#     command_line = 'python ' + path + '/egta_subproc.py'
#     call_and_wait(command_line)
#     print("Done do_train_and_sim")


# load_env: the name of env to be loaded.
# env_name: the name of env to be generated.
# MPI_flag: if running simulation in parallel or not.

# def initialize(load_env=None, env_name=None, MPI_flag = False):
def initialize(load_env=None, env_name=None):
    print("=======================================================")
    print("=======Begin Initialization and first epoch============")
    print("=======================================================")

    # Create Environment
    if isinstance(load_env,str):
        path = os.getcwd() + load_env + '.pkl'
        if not fp.isExist(path):
            raise ValueError("The env being loaded does not exist.")
        env = fp.load_pkl(path)
    else:
        # env is created and saved.
        env = dag.env_rand_gen_and_save(env_name)

    # save graph copy
    env.save_graph_copy()
    env.save_mask_copy()

    # create players and point to their env
    env.create_players()
    env.create_action_space()

    # load param
    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    # initialize game data
    game = game_data.Game_data(env, num_episodes=param['num_episodes'], threshold=param['threshold'])
    game.set_hado_param(param=param['hado_param'])
    game.set_hado_time_step(param['retrain_timesteps'])
    game.env.defender.set_env_belong_to(game.env)
    game.env.attacker.set_env_belong_to(game.env)

    env.defender.set_env_belong_to(env)
    env.attacker.set_env_belong_to(env)

    # uniform strategy has been produced ahead of time
    print("epoch 1:", datetime.datetime.now())
    epoch = 1

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    game.add_att_str(act_att)
    game.add_def_str(act_def)

    print('Begin simulation for uniform strategy.')
    sys.stdout.flush()
    # simulate using random strategies and initialize payoff matrix
    # if MPI_flag:
    #     aReward, dReward = do_MPI_sim(act_att, act_def)
    # else:
    aReward, dReward = series_sim(game.env, game, act_att, act_def, game.num_episodes)
    print('Done simulation for uniform strategy.')
    sys.stdout.flush()

    game.init_payoffmatrix(dReward, aReward)
    ne = {}
    ne[0] = np.array([1], dtype=np.float32)
    ne[1] = np.array([1], dtype=np.float32)
    game.add_nasheq(epoch, ne)

    # save a copy of game data
    game_path = os.getcwd() + '/game_data/game.pkl'
    fp.save_pkl(game, game_path)

    sys.stdout.flush()
    return game

# def EGTA(env, game, start_hado = 2, retrain=False, epoch = 1, game_path = os.getcwd() + '/game_data/game.pkl', MPI_flag = False):
def EGTA(start_hado=2, retrain=False, epoch=1, game_path=os.getcwd() + '/game_data/game.pkl'):

    if retrain:
        print("=======================================================")
        print("==============Begin Running HADO-EGTA==================")
        print("=======================================================")
    else:
        print("=======================================================")
        print("===============Begin Running DO-EGTA===================")
        print("=======================================================")

    sys.stdout.flush()
    arg_path = os.getcwd() + '/inner_egta_arg/'

    hado_arg = (start_hado, retrain)
    epoch_arg = epoch

    fp.save_pkl(hado_arg,path=arg_path+'hado_arg.pkl')
    fp.save_pkl(epoch_arg,path=arg_path+'epoch_arg.pkl')

    count = 18
    while count != 0:
    # while True:
        do_train_and_sim()
        game = fp.load_pkl(game_path)
        epoch = fp.load_pkl(arg_path+'epoch_arg.pkl')
        # find nash equilibrium using gambit analysis
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        print("Begin Gambit analysis.")
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        fp.save_pkl(game, game_path)
        print("Round_" + str(epoch) + " has done and game was saved.")
        print("=======================================================")
        # break
        count -= 1

        sys.stdout.flush() #TODO: make sure this is correct.

    print("END: " + str(epoch))
    os._exit(os.EX_OK)

def EGTA_restart(restart_epoch, start_hado = 2, retrain=False, game_path = os.getcwd() + '/game_data/game.pkl'):

    if retrain:
        print("=======================================================")
        print("============Continue Running HADO-EGTA=================")
        print("=======================================================")
    else:
        print("=======================================================")
        print("=============Continue Running DO-EGTA==================")
        print("=======================================================")

    epoch = restart_epoch - 1

    sys.stdout.flush()
    arg_path = os.getcwd() + '/inner_egta_arg/'

    hado_arg = (start_hado, retrain)
    epoch_arg = epoch

    fp.save_pkl(hado_arg, path=arg_path + 'hado_arg.pkl')
    fp.save_pkl(epoch_arg, path=arg_path + 'epoch_arg.pkl')


    count = 8 - restart_epoch
    while count != 0:
    # while True:
        do_train_and_sim()
        game = fp.load_pkl(game_path)
        epoch = fp.load_pkl(arg_path + 'epoch_arg.pkl')
        #
        # find nash equilibrium using gambit analysis
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        print("Begin Gambit analysis.")
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        game.env.attacker.nn_att = None
        game.env.defender.nn_def = None
        fp.save_pkl(game, game_path)
        print("Round_" + str(epoch) + " has done and game was saved.")
        print("=======================================================")
        # break
        count -= 1

        sys.stdout.flush() #TODO: make sure this is correct.

    print("END EPOCH: " + str(epoch))
    print(datetime.datetime.now())
    # os._exit(os.EX_OK)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    game = initialize(env_name='test_env')
    # EGTA(env, game, retrain=True)
    EGTA(retrain=False)
    # EGTA_restart(restart_epoch=4)








