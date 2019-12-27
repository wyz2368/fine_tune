from attackgraph import json_op as jp
from baselines.common import models
from baselines.deepq.deepq import learn_multi_nets, Learner
import os
import numpy as np
# import copy

DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'

def training_att(game, mix_str_def, epoch, retrain = False):
    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while training")

    # env = copy.deepcopy(game.env)
    print("training_att mix_str_def is ", mix_str_def)

    if not np.all(mix_str_def >= 0):
        print('---------------------------------------------------------------------------')
        print("WARNING: Gambit returns negative mixed strategy! DO-EGTA may have finished.")
        print('---------------------------------------------------------------------------')

        mix_str_def[mix_str_def < 0]=0

    if np.sum(mix_str_def) != 1:
        print("Sum is corrected to 1.")
        mix_str_def = mix_str_def/np.sum(mix_str_def)

    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if retrain:
        scope = 'att_str_retrain' + str(0) + '.pkl' + '/'
    else:
        scope = 'att_str_epoch' + str(epoch) + '.pkl' + '/'

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att, a_BD, harmfulness = learner.learn_multi_nets(
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
            if retrain:
                act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl', 'att_str_retrain' + str(0) + '.pkl' + '/')
            else:
                act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()

    # print("harmfulness:", harmfulness)

    ### Fine-tuning
    env.reset_everything()
    env.set_training_flag(1)
    positive_index = np.where(mix_str_def>0)[0]
    ft_mix_str_def = np.zeros(len(mix_str_def))
    average = []
    for i in positive_index:
        if harmfulness[i] != 1000:
            average.append(harmfulness[i])
    mean = np.mean(average)

    fine_tune_list = []
    for i in positive_index:
        if harmfulness[i] < mean and harmfulness[i] != 1000:
            fine_tune_list.append(i)

    print('fine_tune_list:', fine_tune_list)
    if len(fine_tune_list) > 0 and len(positive_index) != 1:
        print("...Starting fine-tuning...")
        for j in fine_tune_list:
            ft_mix_str_def[j] = 1/len(fine_tune_list)

        print("Fine-tuning mixed strategy is ", ft_mix_str_def)
        env.defender.set_mix_strategy(ft_mix_str_def)
        learner1 = Learner()
        with learner1.graph.as_default():
            with learner1.sess.as_default():

                act_att, _, _ = learner1.learn_multi_nets(
                    env,
                    network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                    lr=param['lr'],
                    total_timesteps= param['retrain_timesteps'],
                    exploration_fraction=param['retrain_exploration'],
                    exploration_final_eps=param['exploration_final_eps'],
                    print_freq=param['print_freq'],
                    param_noise=param['param_noise'],
                    gamma=param['gamma'],
                    prioritized_replay=param['prioritized_replay'],
                    checkpoint_freq=param['checkpoint_freq'],
                    load_path= DIR_att + "att_str_epoch" + str(epoch) + ".pkl",
                    scope=scope,
                    epoch=epoch,
                    fine_tuning=True
                )
                print("Saving attacker's model to pickle.")
                if retrain:
                    act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl',
                                 'att_str_retrain' + str(0) + '.pkl' + '/')
                else:
                    act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl",
                                 'att_str_epoch' + str(epoch) + '.pkl' + '/')
        learner1.sess.close()
        print("Fine Tuning Done")

    return a_BD




def training_def(game, mix_str_att, epoch, retrain = False):
    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while retraining")

    print("training_def mix_str_att is ", mix_str_att)

    if not np.all(mix_str_att >= 0):
        print('---------------------------------------------------------------------------')
        print("WARNING: Gambit returns negative mixed strategy! DO-EGTA may have finished.")
        print('---------------------------------------------------------------------------')
        mix_str_att[mix_str_att < 0] = 0

    if np.sum(mix_str_att) != 1:
        print("Sum is corrected to 1.")
        mix_str_att = mix_str_att/np.sum(mix_str_att)

    # env = copy.deepcopy(game.env)
    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if retrain:
        scope = 'def_str_retrain' + str(0) + '.pkl' + '/'
    else:
        scope = 'def_str_epoch' + str(epoch) + '.pkl' + '/'

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def, d_BD, harmfulness = learner.learn_multi_nets(
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
                scope = scope,
                epoch=epoch
            )
            print("Saving defender's model to pickle.")
            if retrain:
                act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl', 'def_str_retrain' + str(0) + '.pkl' + '/')
            else:
                act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()

    # Fine-tuning
    env.reset_everything()
    env.set_training_flag(0)
    positive_index = np.where(mix_str_att > 0)[0]
    ft_mix_str_att = np.zeros(len(mix_str_att))
    average = []
    for i in positive_index:
        if harmfulness[i] != 0:
            average.append(harmfulness[i])
    mean = np.mean(average)

    fine_tune_list = []
    for i in positive_index:
        if harmfulness[i] < mean and harmfulness[i] != 0:
            fine_tune_list.append(i)


    if len(fine_tune_list) > 1 and len(positive_index) != 1:
        print("...Starting fine-tuning...")
        for j in fine_tune_list:
            ft_mix_str_att[j] = 1 / len(fine_tune_list)

        print("Fine-tuning mixed strategy is ", ft_mix_str_att)
        env.defender.set_mix_strategy(ft_mix_str_att)
        learner1 = Learner()
        with learner1.graph.as_default():
            with learner1.sess.as_default():
                act_def, _, _ = learner1.learn_multi_nets(
                    env,
                    network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                    lr=param['lr'],
                    total_timesteps=param['retrain_timesteps'],
                    exploration_fraction=param['retrain_exploration'],
                    exploration_final_eps=param['exploration_final_eps'],
                    print_freq=param['print_freq'],
                    param_noise=param['param_noise'],
                    gamma=param['gamma'],
                    prioritized_replay=param['prioritized_replay'],
                    checkpoint_freq=param['checkpoint_freq'],
                    scope=scope,
                    epoch=epoch,
                    fine_tuning=True
                )
                print("Saving defender's model to pickle.")
                if retrain:
                    act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl',
                                 'def_str_retrain' + str(0) + '.pkl' + '/')
                else:
                    act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
        learner1.sess.close()

    return d_BD



# for all strategies learned by retraining, the scope index is 0.
def training_hado_att(game):
    param = game.param
    mix_str_def = game.hado_str(identity=0, param=param)

    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while retraining")

    if np.sum(mix_str_def) != 1:
        print("Sum is corrected to 1.")
        mix_str_def = mix_str_def/np.sum(mix_str_def)

    # env = copy.deepcopy(game.env)
    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner(retrain=True, freq=param['retrain_freq'])
    # TODO: add epoch???
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att, _ = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =param['lr'],
                total_timesteps=param['retrain_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'att_str_retrain' + str(0) + '.pkl' + '/',
                load_path=os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl'
            )
            # print("Saving attacker's model to pickle.")
            # act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()


def training_hado_def(game):
    param = game.param
    mix_str_att = game.hado_str(identity=1, param=param)

    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while training")

    # env = copy.deepcopy(game.env)
    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner(retrain=True, freq=param['retrain_freq'])
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def, _ = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=param['lr'],
                total_timesteps=param['retrain_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'def_str_retrain' + str(0) + '.pkl' + '/',
                load_path = os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl'
            )
            # print("Saving defender's model to pickle.")
            # act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()