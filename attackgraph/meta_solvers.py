import numpy as np

def mixed_ne(game, epoch):
    mix_str_def = np.zeros(epoch)
    mix_str_att = np.zeros(epoch)
    for i in np.arange(1, epoch+1):
        temp = game.nasheq[i][0].copy()
        mix_str_def[:len(temp)] += temp
        temp = game.nasheq[i][1].copy()
        mix_str_att[:len(temp)] += temp
    mix_str_def = mix_str_def/np.sum(mix_str_def)
    mix_str_att = mix_str_att/np.sum(mix_str_att)
    return mix_str_def, mix_str_att

def weighted_ne(game, epoch, gamma):
    # gamma is the discount factor for NEs.
    #TODO: Be careful about the normalization and the value of gamma.
    mix_str_def = np.zeros(epoch)
    mix_str_att = np.zeros(epoch)
    for i in np.arange(1, epoch+1):
        temp = game.nasheq[i][0].copy()
        mix_str_def[:len(temp)] += temp * gamma**(epoch-i)
        temp = game.nasheq[i][1].copy()
        mix_str_att[:len(temp)] += temp * gamma**(epoch-i)
    mix_str_def = mix_str_def / np.sum(mix_str_def)
    mix_str_att = mix_str_att / np.sum(mix_str_att)

    return mix_str_def, mix_str_att


def mixed_ne_finite_mem(game, epoch, mem_size):
    mix_str_def = np.zeros(epoch)
    mix_str_att = np.zeros(epoch)
    if mem_size >= epoch:
        for i in np.arange(1, epoch + 1):
            temp = game.nasheq[i][0].copy()
            mix_str_def[:len(temp)] += temp
            temp = game.nasheq[i][1].copy()
            mix_str_att[:len(temp)] += temp
    else:
        for i in np.arange(epoch-mem_size+1, epoch + 1):
            temp = game.nasheq[i][0].copy()
            mix_str_def[:len(temp)] += temp
            temp = game.nasheq[i][1].copy()
            mix_str_att[:len(temp)] += temp
    mix_str_def = mix_str_def / np.sum(mix_str_def)
    mix_str_att = mix_str_att / np.sum(mix_str_att)

    return mix_str_def, mix_str_att

def weight_ne_finite_mem(game, epoch, gamma, mem_size):
    mix_str_def = np.zeros(epoch)
    mix_str_att = np.zeros(epoch)
    if mem_size >= epoch:
        for i in np.arange(1, epoch + 1):
            temp = game.nasheq[i][0].copy()
            mix_str_def[:len(temp)] += temp * gamma ** (epoch - i)
            temp = game.nasheq[i][1].copy()
            mix_str_att[:len(temp)] += temp * gamma ** (epoch - i)
    else:
        for i in np.arange(epoch-mem_size+1, epoch + 1):
            temp = game.nasheq[i][0].copy()
            mix_str_def[:len(temp)] += temp * gamma ** (epoch - i)
            temp = game.nasheq[i][1].copy()
            mix_str_att[:len(temp)] += temp * gamma ** (epoch - i)
    mix_str_def = mix_str_def / np.sum(mix_str_def)
    mix_str_att = mix_str_att / np.sum(mix_str_att)

    return mix_str_def, mix_str_att

#TODO: eps could vary w.r.t str.
def regret_matching_vs_mean(game, epoch, eps, gamma):
    if epoch == 1:
        mix_str_def = game.nasheq[epoch][0]
        mix_str_att = game.nasheq[epoch][1]
        return mix_str_def, mix_str_att

    num_str = len(game.att_str)
    if num_str - len(game.str_regret_att) != 1:
        raise ValueError("Length of str_regret_att does not match num_str")
    game.str_regret_att = np.append(game.str_regret_att, 0)
    game.str_regret_def = np.append(game.str_regret_def, 0)

    str_regret_att = game.str_regret_att.copy()
    str_regret_def = game.str_regret_def.copy()
    payoff_def, payoff_att = mean_payoff(game, str_regret_att, str_regret_def)

    str_regret_def = np.reshape(str_regret_def, newshape=(num_str,1))
    if num_str == 2:
        game.str_regret_att[0] = 0
        game.str_regret_def[0] = 0

    #TODO: make sure the sign of regret.
    regret_vec_def = np.sum(game.payoffmatrix_def * str_regret_att, axis=1) - payoff_def
    regret_vec_att = np.sum(str_regret_def * game.payoffmatrix_att, axis=0) - payoff_att
    regret_vec_def = np.reshape(regret_vec_def, newshape=(num_str,))

    game.str_regret_att = np.maximum(game.str_regret_att * gamma + regret_vec_att * (1-gamma), eps)
    game.str_regret_def = np.maximum(game.str_regret_def * gamma + regret_vec_def * (1-gamma), eps)

    #TODO: Be careful about the normalization.
    #Due to the existence of eps, np.sum(game.str_regret_att)!=0.
    mix_str_att = game.str_regret_att/np.sum(game.str_regret_att)
    mix_str_def = game.str_regret_def/np.sum(game.str_regret_def)

    for i, item in enumerate(mix_str_att):
        if item < 0.05:
            mix_str_att[i] = 0

    for i, item in enumerate(mix_str_def):
        if item < 0.05:
            mix_str_def[i] = 0

    mix_str_def = mix_str_def / np.sum(mix_str_def)
    mix_str_att = mix_str_att / np.sum(mix_str_att)

    return mix_str_def, mix_str_att


def regret_matching_emax(game, epoch, eps):
    if epoch == 1:
        mix_str_def = game.nasheq[epoch][0]
        mix_str_att = game.nasheq[epoch][1]
        return mix_str_def, mix_str_att

    num_str = len(game.att_str)
    if len(game.str_regret_att) >= num_str:
        raise ValueError("Length of str_regret_att does not match num_str")
    game.str_regret_att = np.append(game.str_regret_att, 0)
    game.str_regret_def = np.append(game.str_regret_def, 0)

    str_regret_att = game.str_regret_att.copy()
    str_regret_def = game.str_regret_def.copy()
    payoff_def, payoff_att = payoff(game, str_regret_att, str_regret_def)

    str_regret_def = np.reshape(str_regret_def, newshape=(num_str,1))
    if num_str == 2:
        game.str_regret_att[0] = 0
        game.str_regret_def[0] = 0

    #TODO: make sure the sign of regret.
    regret_vec_def = np.sum(game.payoffmatrix_def*str_regret_att, axis=1) - payoff_def
    regret_vec_att = np.sum(str_regret_def*game.payoffmatrix_att, axis=0) - payoff_att
    regret_vec_def = np.reshape(regret_vec_def, newshape=(num_str,))

    game.str_regret_att = np.maximum(game.str_regret_att + regret_vec_att, eps)
    game.str_regret_def = np.maximum(game.str_regret_def + regret_vec_def, eps)

    #TODO: Be careful about the normalization.
    mix_str_att= game.str_regret_att/np.sum(game.str_regret_att)
    mix_str_def = game.str_regret_def/np.sum(game.str_regret_def)

    return mix_str_def, mix_str_att

def payoff(game, nash_att, nash_def):
    num_str = len(nash_def)
    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.sum(nash_def * game.payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * game.payoffmatrix_att * nash_att), decimals=2)

    return dPayoff, aPayoff

def mean_payoff(game, nash_att, nash_def):
    num_str = len(nash_def)
    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.mean(np.sum(game.payoffmatrix_def * nash_att, axis=1)), decimals=2)
    aPayoff = np.round(np.mean(np.sum(nash_def * game.payoffmatrix_att, axis=0)), decimals=2)

    return dPayoff, aPayoff


def adaptive():
    pass









