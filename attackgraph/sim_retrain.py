# from attackgraph.sim_MPI_retrain import do_MPI_sim_retrain
from attackgraph.simulation import series_sim_retrain
from attackgraph import file_op as fp
import os
import numpy as np
import joblib


#TODO: sim_MPI may cause error since name==main os.exit
def sim_retrain(env, game, mix_str_att, mix_str_def, epoch):
    # sim for retained attacker
    print("Begin sim_retrain_att.")
    a_BD = sim_retrain_att(env, game, mix_str_def, epoch)
    print("Done sim_retrain_att.")
    # sim for retained defender
    print('Begin sim_retrain_def')
    d_BD = sim_retrain_def(env, game, mix_str_att, epoch)
    print("Done sim_retrain_def")

    return a_BD, d_BD


def sim_retrain_att(env, game, mix_str_def,  epoch):
    rewards_att = fp.load_pkl(os.getcwd() + '/retrained_rew/' + 'rewards_att.pkl') # reward is np.array([1,2,3,4])
    k, gamma, alpha = game.param
    DIR = os.getcwd() + '/retrain_att/'
    str_list = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and '.pkl' in name]
    num_str = len(str_list)
    if num_str != len(rewards_att):
        print('***************************')
        print('Retrain reward length does not match!')
        print('***************************')
        raise ValueError('Retrain reward length does not match!')
    util = []
    for i in range(num_str):
        nn_att = 'att_str_retrain' + str(i) + ".pkl"
        nn_def = mix_str_def
        # if MPI_flag:
        #     a_BD, _ = do_MPI_sim_retrain(nn_att, nn_def)
        # else:
        a_BD, _ = series_sim_retrain(env, game, nn_att, nn_def, 10)

        util.append(alpha*a_BD+(1-alpha)*rewards_att[i])

    best_idx = np.argmax(np.array(util))
    os.rename(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(best_idx) + ".pkl", os.getcwd() + "/attacker_strategies/" + 'att_str_epoch' + str(epoch) + '.pkl')
    change_scope(path=os.getcwd() + "/attacker_strategies/" + 'att_str_epoch' + str(epoch) + '.pkl', epoch=epoch, identity=1)
    return np.max(np.array(util))



def sim_retrain_def(env, game, mix_str_att,  epoch):
    rewards_def = fp.load_pkl(os.getcwd() + '/retrained_rew/' + 'rewards_def.pkl')
    k, gamma, alpha = game.param
    DIR = os.getcwd() + '/retrain_def/'
    str_list = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and '.pkl' in name]
    num_str = len(str_list)
    if num_str != len(rewards_def):
        print('***************************')
        print('Retrain reward length does not match!')
        print('***************************')
        raise ValueError('Retrain reward length does not match!')
    util = []
    for i in range(num_str):
        nn_att = mix_str_att
        nn_def = "def_str_retrain" + str(i) + ".pkl"
        # if MPI_flag:
        #     _, d_BD = do_MPI_sim_retrain(nn_att, nn_def)
        # else:
        _, d_BD = series_sim_retrain(env, game, nn_att, nn_def, 10)

        util.append(alpha * d_BD + (1 - alpha) * rewards_def[i])

    best_idx = np.argmax(np.array(util))
    os.rename(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(best_idx) + ".pkl", os.getcwd() + "/defender_strategies/" + 'def_str_epoch' + str(epoch) + '.pkl')
    change_scope(path=os.getcwd() + "/defender_strategies/" + 'def_str_epoch' + str(epoch) + '.pkl', epoch=epoch, identity=0)
    return np.max(np.array(util))


def change_scope(path, epoch, identity):
    loaded_params = joblib.load(os.path.expanduser(path))
    new_params = {}
    keys = loaded_params.keys()

    if identity == 0:
        old_keys = 'def_str_retrain0'
        new_keys = 'def_str_epoch' + str(epoch)
    elif identity == 1:
        old_keys = 'att_str_retrain0'
        new_keys = 'att_str_epoch' + str(epoch)
    else:
        raise ValueError("Identity error!")

    for key in keys:
        a = key.replace(old_keys, new_keys)
        new_params[a] = loaded_params[key]

    joblib.dump(new_params, path)















