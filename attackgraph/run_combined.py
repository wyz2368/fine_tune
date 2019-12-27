import sys
sys.path.append('/home/wangyzh/combine_run_init')

from attackgraph.simulator_combined import scan_and_sim
from attackgraph.restart_combined import scan_and_sim_restart
from attackgraph.evaluation_combined import do_evaluation
import datetime
import warnings
import os



# TODO: dictionary is not ordered. So check the logic.
def run(method_list):
    done_list = ["reward_shaping", "fictitious"]
    payoff_matrix_att, payoff_matrix_def, child_partition = scan_and_sim_restart(method_list, done_list)
    # payoff_matrix_att, payoff_matrix_def, child_partition = scan_and_sim(method_list)
    # do_evaluation(payoff_matrix_def, payoff_matrix_att, child_partition)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print("...Begin Payoff Matrix Simulation")
    print("Current time is ", datetime.datetime.now())
    method_list = ["reward_shaping", "fictitious", 'weighted', 'regret_matching', 'BR_fic', 'BR_weighted', 'BR_selfplay']
    run(method_list)
    print("Current time is ", datetime.datetime.now())
    print("...Done Payoff Matrix Simulation...")