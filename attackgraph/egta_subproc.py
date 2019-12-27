from attackgraph.subproc import call_and_wait

# Packages import
# import numpy as np
import os
import datetime
# import sys
# import psutil

# Modules import

from attackgraph import file_op as fp
# from attackgraph import json_op as jp
from attackgraph import sim_Series
from attackgraph import training
from attackgraph import util
# from attackgraph.simulation import series_sim
from attackgraph.sim_retrain import sim_retrain

def do_train_and_sim():
    print('Begin do_train_and_sim')
    path = os.getcwd()
    command_line = 'python ' + path + '/egta_subproc.py'
    call_and_wait(command_line)
    print("Done do_train_and_sim")


def train_and_sim():
    arg_path = os.getcwd() + '/inner_egta_arg/'

    start_hado, retrain = fp.load_pkl(arg_path+'hado_arg.pkl')
    epoch = fp.load_pkl(arg_path+'epoch_arg.pkl')

    game_path = os.getcwd() + '/game_data/game.pkl'
    game = fp.load_pkl(game_path)
    env = game.env

    retrain_start = False

    mix_str_def = game.nasheq[epoch][0]
    mix_str_att = game.nasheq[epoch][1]
    aPayoff, dPayoff = util.payoff_mixed_NE(game, epoch)

    game.att_payoff.append(aPayoff)
    game.def_payoff.append(dPayoff)

    # increase epoch
    epoch += 1
    print("Current epoch is " + str(epoch))
    print("epoch " + str(epoch) + ':', datetime.datetime.now())

    # train and save RL agents

    if retrain and epoch > start_hado:
        retrain_start = True

    print("Begin training attacker......")
    a_BD = training.training_att(game, mix_str_def, epoch, retrain=retrain_start)
    print("Attacker training done......")

    print("Begin training defender......")
    d_BD = training.training_def(game, mix_str_att, epoch, retrain=retrain_start)
    print("Defender training done......")

    if retrain and epoch > start_hado:
        print("Begin retraining attacker......")
        training.training_hado_att(game)
        print("Attacker retraining done......")

        print("Begin retraining defender......")
        training.training_hado_def(game)
        print("Defender retraining done......")

        # Simulation for retrained strategies and choose the best one as player's strategy.
        print('Begin retrained sim......')
        a_BD, d_BD = sim_retrain(env, game, mix_str_att, mix_str_def, epoch)
        print('Done retrained sim......')

    game.att_BD_list.append(a_BD)
    game.def_BD_list.append(d_BD)

    # else:
    #
    #     # Judge beneficial deviation
    #     # one plays nn and another plays ne strategy
    #     print("Simulating attacker payoff. New strategy vs. mixed opponent strategy.")
    #     nn_att = "att_str_epoch" + str(epoch) + ".pkl"
    #     nn_def = mix_str_def
    #     # if MPI_flag:
    #     #     a_BD, _ = do_MPI_sim(nn_att, nn_def)
    #     # else:
    #     a_BD, _ = series_sim(env, game, nn_att, nn_def, game.num_episodes)
    #     print("Simulation done for a_BD.")
    #
    #     print("Simulating defender's payoff. New strategy vs. mixed opponent strategy.")
    #     nn_att = mix_str_att
    #     nn_def = "def_str_epoch" + str(epoch) + ".pkl"
    #     # if MPI_flag:
    #     #     _, d_BD = do_MPI_sim(nn_att, nn_def)
    #     # else:
    #     _, d_BD = series_sim(env, game, nn_att, nn_def, game.num_episodes)
    #     print("Simulation done for d_BD.")

    # #TODO: This may lead to early stop.
    # if a_BD - aPayoff < game.threshold and d_BD - dPayoff < game.threshold:
    #     print("*************************")
    #     print("aPayoff=", aPayoff, " ", "dPayoff=", dPayoff)
    #     print("a_BD=", a_BD, " ", "d_BD=", d_BD)
    #     print("*************************")
    #     break
    #
    game.add_att_str("att_str_epoch" + str(epoch) + ".pkl")
    game.add_def_str("def_str_epoch" + str(epoch) + ".pkl")

    # simulate and extend the payoff matrix.
    # game = sim_Series.sim_and_modifiy_Series_with_game(game, MPI_flag=MPI_flag)
    game = sim_Series.sim_and_modifiy_Series_with_game(game)

    game.env.attacker.nn_att = None
    game.env.defender.nn_def = None

    print('a_BD_list', game.att_BD_list)
    print('aPayoff', game.att_payoff)
    print('d_BD_list', game.def_BD_list)
    print('dPayoff', game.def_payoff)

    fp.save_pkl(game, game_path)
    fp.save_pkl(epoch, arg_path+'epoch_arg.pkl')

if __name__ == '__main__':
    train_and_sim()