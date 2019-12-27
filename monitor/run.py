import numpy as np
from monitor.screening import screen
from attackgraph.ne_search import whole_payoff_matrix
from attackgraph import gambit_analysis as ga


def create_paths(methods_list, base_path = '/home/wangyzh/'):
    """
    create paths to files that contain strategies.
    :param methods_list: a list of names of heuristics.
    :param base_path: path of the home directory.
    :return:
    """
    paths = {}
    for method in methods_list:
        paths[method] = base_path + method
    return paths

def find_heuristic_position(child_partition):
    """
    This function finds the starting position of different heuristics in a combined game payoff matrix.
    :param child_partition:
    :return:
    """
    position = {}
    i = 0
    for method in child_partition:
        position[method] = (i, i+child_partition[method])
        i += child_partition[method]
    return position

def scan_and_sim(methods_list):
    """
    This fuction first
    :param methods_list:
    :return:
    """
    paths = create_paths(methods_list)
    str_dict_def, str_dict_att = screen(paths)

    # create children partition
    child_partition = {}
    total_num_str = 0
    for method in methods_list:
        child_partition[method] = len(str_dict_def[method])
        total_num_str += child_partition[method]

    # create strategy path dictionary
    str_path_dict = {}
    str_path_dict[0] = {}
    str_path_dict[1] = {}

    att_path = '/attackgraph/attacker_strategies/'
    def_path = '/attackgraph/defender_strategies/'

    for method in methods_list:
        str_path_dict[0][method] = paths[method] + def_path
        str_path_dict[1][method] = paths[method] + att_path

    payoff_matrix_att, payoff_matrix_def = whole_payoff_matrix(total_num_str,child_partition,str_path_dict=str_path_dict)

    return payoff_matrix_att, payoff_matrix_def, child_partition

def partial_matrix_ne_search(payoff_matrix_att, payoff_matrix_def, child_partition):
    ne_dict = {}
    heuristic_pos = find_heuristic_position(child_partition)
    for method in child_partition:
        ne_dict[method] = {}
        # find the position of heuristic.
        h_pos = heuristic_pos[method]

        # find the NE of the partial matrix.
        nash_att, _ = ga.do_gambit_analysis(payoff_matrix_def, payoff_matrix_att[h_pos[0], h_pos[1]], maxent=False, minent=False)
        _, nash_def = ga.do_gambit_analysis(payoff_matrix_def[h_pos[0], h_pos[1]], payoff_matrix_att, maxent=False, minent=False)

        # add a zero for uniform strategy.
        ne_dict[method][0] = np.insert(nash_def, 0, 0)
        ne_dict[method][1] = np.insert(nash_att, 0 ,0)

    return ne_dict

def print_ne_dict(ne_dict):
    print("....Print the NE based on partial payoff matrix....")
    for method in ne_dict:
        print("The current method is ", method)
        print("Defender's NE is ", ne_dict[method][0])
        print("Attacker's NE is ", ne_dict[method][1])
        print("--------------------------------------")

    print("....End of the printing....")


if __name__ == '__main__':
    method_list = ["reward_shaping", "BR_selfplay"]
    payoff_matrix_att, payoff_matrix_def, child_partition = scan_and_sim(method_list)