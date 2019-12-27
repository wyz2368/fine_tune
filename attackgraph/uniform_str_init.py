import numpy as np
from attackgraph import file_op as fp
import os

DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'


def act_att(ob, mask, training_flag, stochastic=True, update_eps=-1):
    if training_flag != 1:
        raise ValueError("training flag for uniform att str is not 1")
    legal_action = np.where(mask[0] == 0)[0]
    return [np.random.choice(legal_action)]


def act_def(ob, mask, training_flag, stochastic=True, update_eps=-1):
    if training_flag != 0:
        raise ValueError("training flag for uniform def str is not 0")
    legal_action = np.where(mask[0] == 0)[0]
    return [np.random.choice(legal_action)]


fp.save_pkl(act_att, DIR_att + "att_str_epoch" + str(1) + ".pkl")
fp.save_pkl(act_def, DIR_def + "def_str_epoch" + str(1) + ".pkl")