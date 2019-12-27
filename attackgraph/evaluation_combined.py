import numpy as np
from attackgraph.gambit_analysis import do_gambit_analysis
from attackgraph import file_op as fp
from attackgraph.utils_combined import create_paths, find_heuristic_position
import os


def find_all_NE(payoffmatrix_def, payoffmatrix_att):
    # Paired NE.
    # nash_att_list = [np.array([0.5,0.2,0.3]), ...]
    # nash_def_list = [np.array([0.5,0.2,0.3]), ...]
    nash_att_list, nash_def_list = do_gambit_analysis(payoffmatrix_def, payoffmatrix_att, return_list=True)
    return nash_att_list, nash_def_list

def regret(nash_att, nash_def, payoffmatrix_att, payoffmatrix_def):
    """
    Calculate the regret vectors for both players.
    :param nash_att: attacker's NE
    :param nash_def: defender's NE
    :param payoffmatrix_att: attacker's payoff matrix
    :param payoffmatrix_def: defender's payoff matrix
    :return: regret vectors
    """
    num_str = len(nash_att)
    x1, y1 = np.shape(payoffmatrix_def)
    x2, y2 = np.shape(payoffmatrix_att)
    if x1 != y1 or x1 != x2 or x2 != y2 or x1 != num_str:
        raise ValueError("Dim of NE does not match payoff matrix.")

    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.sum(nash_def * payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * payoffmatrix_att * nash_att), decimals=2)

    utils_def = np.round(np.sum(payoffmatrix_def * nash_att, axis=1), decimals=2)
    utils_att = np.round(np.sum(nash_def * payoffmatrix_att, axis=0), decimals=2)

    regret_def = utils_def - dPayoff
    regret_att = utils_att - aPayoff

    regret_def = np.reshape(regret_def, newshape=np.shape(regret_att))

    regret_att = -regret_att
    regret_def = -regret_def

    return regret_att, regret_def

# dummy function
def create_ne_dict(methods_list, newest=False):
    ne_dict ={}
    paths = create_paths(methods_list)
    for method in methods_list:
        path = paths[method] + '/game.pkl'
        game = fp.load_pkl(path)
        if newest:
            idx = len(game.nasheq.keys())
            nasheq = game.nasheq[idx]
        else:
            nasheq = game.nasheq
        ne_dict[method] = nasheq
    return ne_dict

def NE_regret(regret_vect_att, regret_vect_def, payoffmatrix_att, payoffmatrix_def, child_partition):
    """
    Calculate the regret of each heuristic with respect to the combined game. The strategies of each heuristic only\
    include those in the NE of each heuristic.
    :param regret_vect: regret vector calculated from combined game.
    :param ne_dict: {"baseline": {0: np.array([1,0,1,0...]), 1: np.array([1,0,1,0...])},
    "RS": np.array([0,0,1,0...])} when a strategy is in a NE, that strategy is indicated by 1.
    :return:
    """

    regret_dict = {}
    positions = find_heuristic_position(child_partition)
    for method in child_partition:
        start, end = positions[method]
        print(start, end)
        submatrix_att = payoffmatrix_att[start:end, start:end]
        submatrix_def = payoffmatrix_def[start:end, start:end]

        # submatrix_att = payoffmatrix_att[start:start+30, start:start+30]
        # submatrix_def = payoffmatrix_def[start:start+30, start:start+30]

        nash_att, nash_def = do_gambit_analysis(submatrix_def, submatrix_att, maxent=True)

        nash_att[nash_att>0] = 1
        nash_def[nash_def>0] = 1

        regret_dict[method] = {0: np.sum(regret_vect_def[start:end] * nash_def)/np.sum(nash_def),
                               1: np.sum(regret_vect_att[start:end] * nash_att)/np.sum(nash_att)}

        # regret_dict[method] = {0: np.sum(regret_vect_def[start:start+30] * nash_def) / np.sum(nash_def),
        #                        1: np.sum(regret_vect_att[start:start+30] * nash_att) / np.sum(nash_att)}

    return regret_dict

def regret_fixed_matrix(payoffmatrix_def, payoffmatrix_att, child_partition):
    positions = find_heuristic_position(child_partition)
    for method in child_partition:
        start, end = positions[method]
        print(start, end)
        submatrix_att = payoffmatrix_att[start:end, start:end]
        submatrix_def = payoffmatrix_def[start:end, start:end]

        # submatrix_att = payoffmatrix_att[start:start+40, start:start+40]
        # submatrix_def = payoffmatrix_def[start:start+40, start:start+40]

        nash_att, nash_def = do_gambit_analysis(submatrix_def, submatrix_att, maxent=True)

        nash_def = np.reshape(nash_def, newshape=(len(nash_def), 1))

        ne_payoff_def = np.sum(nash_def * submatrix_def * nash_att)
        ne_payoff_att = np.sum(nash_def * submatrix_att * nash_att)

        dev_def = np.max(np.sum(payoffmatrix_def[:, start:end] * nash_att, axis=1))
        dev_att = np.max(np.sum(nash_def * payoffmatrix_att[start:end, :], axis=0))

        # dev_def = np.max(np.sum(payoffmatrix_def[:, start:start+40] * nash_att, axis=1))
        print(np.argmax(np.sum(payoffmatrix_def[:, start:end] * nash_att, axis=1)))
        # dev_att = np.max(np.sum(nash_def * payoffmatrix_att[start:start+40, :], axis=0))
        print(np.argmax(np.sum(nash_def * payoffmatrix_att[start:end, :], axis=0)))

        print('------------------------------------------')
        print("The current method is ", method)
        print("The defender's regret is", np.maximum(dev_def-ne_payoff_def, 0))
        print("The attacker's regret is", np.maximum(dev_att-ne_payoff_att, 0))
    print("==================================================")

def formal_regret_curves(payoffmatrix_def, payoffmatrix_att, child_partition):
    positions = find_heuristic_position(child_partition)
    curves_dict_def = {}
    curves_dict_att = {}
    for method in child_partition:
        curves_dict_def[method] = []
        curves_dict_att[method] = []
    for epoch in np.arange(80):
        for method in child_partition:
            start, end = positions[method]
            print(start, end)

            submatrix_att = payoffmatrix_att[start:start+epoch+1, start:start+epoch+1]
            submatrix_def = payoffmatrix_def[start:start+epoch+1, start:start+epoch+1]

            nash_att, nash_def = do_gambit_analysis(submatrix_def, submatrix_att, maxent=True)

            nash_def = np.reshape(nash_def, newshape=(len(nash_def), 1))

            ne_payoff_def = np.sum(nash_def * submatrix_def * nash_att)
            ne_payoff_att = np.sum(nash_def * submatrix_att * nash_att)


            dev_def = np.max(np.sum(payoffmatrix_def[:, start:start+epoch+1] * nash_att, axis=1))
            dev_att = np.max(np.sum(nash_def * payoffmatrix_att[start:start+epoch+1, :], axis=0))

            curves_dict_def[method].append(np.maximum(dev_def-ne_payoff_def, 0))
            curves_dict_att[method].append(np.maximum(dev_att-ne_payoff_att, 0))

    return curves_dict_def, curves_dict_att


# Measure the regret of subgames during strategy exploration.
# TODO: regret curves should be separate for the defender and the attacker due to the asymmetry.
def regret_curves(payoffmatrix_def, payoffmatrix_att, child_partition):
    """
    Calculate the epsilon of each subgame.
    :param ne_dict: {"baseline": game.nasheq}
    :return:
    """
    curves_att = {}
    curves_def = {}
    num_str, _ = np.shape(payoffmatrix_att)
    positions = find_heuristic_position(child_partition)
    for method in child_partition:
        curves_att[method] = []
        curves_def[method] = []
        start, end = positions[method]
        submatrix_def = payoffmatrix_def[start:end, :]
        submatrix_att = payoffmatrix_att[:, start:end]
        subgame_def = payoffmatrix_def[start:end, start:end]
        subgame_att = payoffmatrix_att[start:end, start:end]

        zeros = np.zeros(end-start)
        for epoch in np.arange(end):
            subsubgame_def = subgame_def[:epoch, :epoch]
            subsubgame_att = subgame_att[:epoch, :epoch]

            # TODO: Error: line 4:2: Expecting outcome or payoff
            nash_att, nash_def = do_gambit_analysis(subsubgame_def, subsubgame_att, maxent=True)

            nash_def = zeros[len(nash_def)] + nash_def
            nash_att = zeros[len(nash_att)] + nash_att

            nash_def = np.reshape(nash_def, newshape=(len(nash_def), 1))

            payoff_vect_att = np.sum(nash_def * submatrix_def, axis=0)
            payoff_vect_def = np.sum(submatrix_att * nash_att, axis=1)

            payoffmatrix_def = np.reshape(payoffmatrix_def, newshape=np.shape(payoff_vect_att))

            nash_payoff_att = np.round(np.sum(nash_def * subgame_att * nash_att), decimals=2)
            nash_payoff_def = np.round(np.sum(nash_def * subgame_def * nash_att), decimals=2)

            deviation_att = np.max(payoff_vect_att)
            deviation_def = np.max(payoff_vect_def)
            regret_att = np.maximum(deviation_att - nash_payoff_att, 0)
            regret_def = np.maximum(deviation_def - nash_payoff_def, 0)

            curves_att[method].append(regret_att)
            curves_def[method].append(regret_def)

    return curves_att, curves_def

# Replaceability Implementation.
def replaceability(nash_att, nash_def, payoffmatrix_def, payoffmatrix_att, child_partition):
    """
    This function calculates the replaceability of heuristics.
    :param child_partition:
    :return:
    """
    rep = {}
    positions = find_heuristic_position(child_partition)
    pos_to_method = {y:x for x,y in positions.iteritems()}

    nash_indicator_att = nash_att.copy()
    nash_indicator_att[nash_indicator_att>0] = 1
    nash_indicator_def = nash_def.copy()
    nash_indicator_def[nash_indicator_def>0] = 1

    dPayoff = np.round(np.sum(nash_def * payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * payoffmatrix_att * nash_att), decimals=2)

    utils_def = np.round(np.sum(payoffmatrix_def * nash_att, axis=1), decimals=2)
    utils_att = np.round(np.sum(nash_def * payoffmatrix_att, axis=0), decimals=2)

    utils_def = np.reshape(utils_def, newshape=np.shape(utils_att))

    for method in child_partition:
        start, end = positions[method]

        utils_def[start:end] = -10000
        utils_att[start:end] = -10000


def vector_convertor(ne_dict):
    '''
    Convert a NE to a indicator vector which indicates which strategy is in the support.
    :param ne_dict: {0:[0.3,0.2,0.5],1:[0.3,0.2,0.5]}
    :return:
    '''
    ones_def = ne_dict[0]
    ones_att = ne_dict[1]

    ones_def[ones_def>0] = 1
    ones_att[ones_att>0] = 1

    return ones_att, ones_def

def do_evaluation(payoffmatrix_def, payoffmatrix_att, child_partition):
    regret_fixed_matrix(payoffmatrix_def, payoffmatrix_att, child_partition)
    nash_att_list, nash_def_list = find_all_NE(payoffmatrix_def, payoffmatrix_att)
    # print("The number of NE is ", len(nash_def_list))
    nash = zip(nash_att_list, nash_def_list)
    regret_dict_list = []
    for nash_att, nash_def in nash:
        # print(nash_att[:81], '\n', nash_att[81:])
        # print(nash_def[:81], '\n', nash_def[81:])
        regret_att, regret_def = regret(nash_att, nash_def, payoffmatrix_att, payoffmatrix_def)
        regret_dict = NE_regret(regret_att, regret_def, payoffmatrix_att, payoffmatrix_def, child_partition)
        regret_dict_list.append(regret_dict)
        for method in regret_dict:
            print('------------------------------------------')
            print("The current method is ", method)
            print("The defender's regret is", regret_dict[method][0])
            print("The attacker's regret is", regret_dict[method][1])
        print("==================================================")



    # save_path = os.getcwd() + '/combined_game/data/'
    save_path = os.getcwd() + '/drawing/matrix/'
    fp.save_pkl(regret_dict_list, save_path + 'regret_dict_list.pkl')

    # curves_att, curves_def = regret_curves(payoffmatrix_def, payoffmatrix_att, child_partition)
    # print(curves_att)
    # fp.save_pkl(curves_att, save_path + 'curves_att.pkl')
    # fp.save_pkl(curves_def, save_path + 'curves_def.pkl')






























