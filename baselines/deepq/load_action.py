
from baselines.common import models
from baselines.deepq.deepq import learn_multi_nets, Learner, learn
import os
from attackgraph import json_op as jp

#TODO: make sure the path is correct
def load_action(path, scope, game, training_flag):

    env = game.env
    env.set_training_flag(training_flag)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    act = learn_multi_nets(
        env,
        network=models.mlp(num_layers=param['num_layers'], num_hidden=param['num_hidden']),
        total_timesteps=0,
        load_path=path,
        scope=scope + '/'
    )
    return act

def load_action_class(path, scope, game, training_flag):

    env = game.env
    env.set_training_flag(training_flag)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner()
    act = learner.learn_multi_nets(
        env,
        network=models.mlp(num_layers=param['num_layers'], num_hidden=param['num_hidden']),
        total_timesteps=0,
        load_path=path,
        scope=scope + '/'
    )

    return act, learner.sess, learner.graph

def load_action_with_default_sess(path, scope, game, training_flag):

    env = game.env
    env.set_training_flag(training_flag)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    act = learn(
        env,
        network=models.mlp(num_layers=param['num_layers'], num_hidden=param['num_hidden']),
        total_timesteps=0,
        load_path=path,
        scope=scope + '/'
    )
    return act