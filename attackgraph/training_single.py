# Packages import
import numpy as np
import sys
sys.path.append('/home/wangyzh/exp')
import warnings
from attackgraph.deepgraph_runner import initialize

from attackgraph import json_op as jp
from baselines.common import models
from baselines.deepq.deepq import Learner
import os

DIR_def = os.getcwd() + '/trained_str/'
DIR_att = os.getcwd() + '/trained_str/'

def training_att_single(game, str_def, epoch):
    print('Begin attacker training against ' + str_def)
    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    mix_str_def = [str_def]
    env.defender.set_mix_strategy(np.array([1]))
    env.defender.set_str_set(mix_str_def)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)


    scope = 'att_str_epoch' + str(epoch) + '.pkl' + '/'

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att, a_BD = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =param['lr'],
                total_timesteps=param['total_timesteps_att'],
                exploration_fraction=param['exploration_fraction_att'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = scope,
                epoch=epoch
            )
            print("Saving attacker's model to pickle.")
            act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()
    print('Done attacker training against ' + str_def)
    return a_BD


def training_def_single(game, str_att, epoch):
    print('Begin defender training against ' + str_att)
    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    mix_str_att = [str_att]
    env.attacker.set_mix_strategy(np.array([1]))
    env.attacker.set_str_set(mix_str_att)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    scope = 'def_str_epoch' + str(epoch) + '.pkl' + '/'

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def, d_BD = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=param['lr'],
                total_timesteps=param['total_timesteps_def'],
                exploration_fraction=param['exploration_fraction_def'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope=scope,
                epoch=epoch
            )
            print("Saving defender's model to pickle.")
            act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", 'def_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()
    print('Done defender training against ' + str_att)
    return d_BD

def do_training(str, training_id):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")
    game = initialize(load_env='run_env_B', env_name=None)

    print("=======================================================")
    print("=======Begin training against single strategy =========")
    print("=======================================================")
    print("Training flag is :", training_id)
    BD_list = []
    epoch = 15
    for i in np.arange(2, epoch):
        if training_id == 1:
            a_BD = training_att_single(game, str, i)
            BD_list.append(a_BD)
        else:
            d_BD = training_def_single(game, str, i)
            BD_list.append(d_BD)
        print("BD_list:", BD_list)
        sys.stdout.flush()

if __name__ == '__main__':
    # do_training('def_str_epoch38.pkl', 1)
    do_training('att_str_epoch29.pkl', 0)